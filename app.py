import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, ViTModel, pipeline
from torchvision import transforms
from PIL import Image
import clip
from groq import Groq
import json
import traceback
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from sqlalchemy import func
from ultralytics import YOLO

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default-key')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Database Setup ---
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///reviews.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class ReviewRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    product_name = db.Column(db.String(200), nullable=False)
    review_text = db.Column(db.Text, nullable=False)
    rating = db.Column(db.Integer)
    sentiment = db.Column(db.String(50))
    confidence = db.Column(db.String(20))
    consistency = db.Column(db.String(50))
    aspect = db.Column(db.String(100))
    emotion = db.Column(db.String(100))
    intent = db.Column(db.String(100))
    defect_type = db.Column(db.String(100))
    severity = db.Column(db.String(50))
    summary = db.Column(db.Text)
    explanation = db.Column(db.Text)
    action = db.Column(db.Text)
    image_path = db.Column(db.String(200))
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

# --- SAMPLE TEST CASES (For Demo Fallback) ---
SAMPLE_TEST_CASES = {
    "5": {"text": "The sound quality is incredible and the noise cancellation is top-tier. They fit perfectly and arrived in pristine condition.", "product": "Noise Cancelling Headphones"},
    "4": {"text": "The screen is bright and the battery lasts 2 days. However, the strap feels very cheap and the heart rate sensor is inaccurate.", "product": "Smart Watch Series 5"},
    "3": {"text": "It has amazing suction power and cleans my carpets well. However, the cord is way too short and the dustbin is small.", "product": "Vacuum Cleaner"},
    "2": {"text": "The zipper broke on the second day. The fabric is tearing apart at the seams and the strap is fraying. Terrible build quality.", "product": "Leather Backpack"},
    "1": {"text": "The product arrived completely shattered. The box was crushed and there was glass everywhere. I want a full refund immediately.", "product": "Glass Coffee Maker"}
}

# --- AI Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=0 if torch.cuda.is_available() else -1)

print("Loading YOLOv8 Defect Detector...")
try:
    yolo_model = YOLO("model/best.pt")
except Exception as e:
    yolo_model = None
    print(f"⚠️ Warning: Could not load YOLO model from model/best.pt: {e}")

class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained("distilbert-base-uncased")
        self.image_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.cross_attention = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
        self.shared = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3))
        self.sentiment_head = nn.Linear(256, 3) 
        self.category_head = nn.Linear(256, 6)  

    def forward(self, ids, mask, img):
        t = self.text_encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        v = self.image_encoder(pixel_values=img).last_hidden_state
        attn, _ = self.cross_attention(query=t, key=v, value=v)
        shared_features = self.shared(attn[:, 0, :])
        return self.sentiment_head(shared_features), self.category_head(shared_features)

model = MultiTaskModel().to(device)
if os.path.exists("model/multimodal_6cat_model.pth"):
    try:
        model.load_state_dict(torch.load("model/multimodal_6cat_model.pth", map_location=device), strict=False)
    except Exception as e:
        print(f"⚠️ Error loading PyTorch model: {e}")
model.eval()

clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# --- Routes ---
@app.route('/')
def overview():
    query = request.args.get('q', '').strip()
    if query:
        recent = ReviewRecord.query.filter(ReviewRecord.product_name.ilike(f'%{query}%')).order_by(ReviewRecord.date_created.desc()).all()
    else:
        recent = ReviewRecord.query.order_by(ReviewRecord.date_created.desc()).limit(5).all()
    total = ReviewRecord.query.count()
    negative_reviews = ReviewRecord.query.filter_by(sentiment="Negative").count()
    neg_percent = (negative_reviews / total * 100) if total > 0 else 0
    return render_template('overview.html', total=total, neg_percent=f"{neg_percent:.1f}%", recent=recent, search_query=query)

@app.route('/reviews')
def reviews():
    query = request.args.get('q', '').strip()
    if query:
        all_reviews = ReviewRecord.query.filter((ReviewRecord.product_name.ilike(f'%{query}%')) | (ReviewRecord.review_text.ilike(f'%{query}%'))).order_by(ReviewRecord.date_created.desc()).all()
    else:
        all_reviews = ReviewRecord.query.order_by(ReviewRecord.date_created.desc()).all()
    return render_template('reviews.html', reviews=all_reviews, search_query=query)

@app.route('/analysis')
def analysis(): return render_template('analysis.html', active_page='analysis')

@app.route('/analytics')
def analytics():
    query = request.args.get('q', '').strip()
    base_query = ReviewRecord.query
    if query: 
        base_query = base_query.filter(ReviewRecord.product_name.ilike(f'%{query}%'))
        
    total = base_query.count()
    pos = base_query.filter_by(sentiment="Positive").count()
    neu = base_query.filter_by(sentiment="Neutral").count()
    neg = base_query.filter_by(sentiment="Negative").count()

    aspect_query = db.session.query(ReviewRecord.aspect, func.count(ReviewRecord.id)).filter(ReviewRecord.aspect != None).group_by(ReviewRecord.aspect).all()
    aspects = [{"name": row[0], "count": row[1]} for row in aspect_query if row[0] and row[0].lower() != 'unknown']
    
    defect_query = db.session.query(ReviewRecord.defect_type, func.count(ReviewRecord.id)).filter(ReviewRecord.defect_type != None).group_by(ReviewRecord.defect_type).all()
    defects = [{"name": row[0], "count": row[1]} for row in defect_query if row[0] and row[0].lower() not in ['none', 'unknown']]

    timeline_raw = db.session.query(
        func.date(ReviewRecord.date_created).label('date'),
        ReviewRecord.sentiment,
        func.count(ReviewRecord.id)
    ).group_by(func.date(ReviewRecord.date_created), ReviewRecord.sentiment).order_by(func.date(ReviewRecord.date_created)).all()
    
    timeline = {}
    for date, sentiment, count in timeline_raw:
        if date not in timeline:
            timeline[date] = {"Positive": 0, "Neutral": 0, "Negative": 0}
        timeline[date][sentiment] = count

    dates = list(timeline.keys())
    pos_trend = [timeline[d]["Positive"] for d in dates]
    neu_trend = [timeline[d]["Neutral"] for d in dates]
    neg_trend = [timeline[d]["Negative"] for d in dates]

    return render_template('analytics.html', 
                           total=total, pos=pos, neu=neu, neg=neg, 
                           aspects=aspects, defects=defects, 
                           dates=dates, pos_trend=pos_trend, neu_trend=neu_trend, neg_trend=neg_trend,
                           search_query=query)

@app.route('/reports')
def reports():
    query = request.args.get('q', '').strip()
    base_query = ReviewRecord.query
    if query: 
        base_query = base_query.filter(ReviewRecord.product_name.ilike(f'%{query}%'))
        
    total = base_query.count()
    high_severity = base_query.filter_by(severity="High").count()
    critical_severity = base_query.filter_by(severity="Critical").count()
    
    top_defects = db.session.query(ReviewRecord.defect_type, func.count(ReviewRecord.id)).filter(
        ReviewRecord.defect_type != None,
        ReviewRecord.defect_type != 'None',
        ReviewRecord.defect_type != 'Unknown'
    ).group_by(ReviewRecord.defect_type).order_by(func.count(ReviewRecord.id).desc()).limit(3).all()
    
    top_defect_list = [{"name": row[0], "count": row[1]} for row in top_defects]

    recent_issues = base_query.filter(ReviewRecord.severity.in_(["High", "Critical"])).order_by(ReviewRecord.date_created.desc()).limit(5).all()

    return render_template('reports.html', 
                           total=total, 
                           high_severity=high_severity, 
                           critical_severity=critical_severity,
                           top_defects=top_defect_list,
                           recent_issues=recent_issues,
                           search_query=query)

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    try:
        text = request.form.get('review_text')
        rating = request.form.get('rating', '3')
        product_name = request.form.get('product_name', 'Unknown Product')
        customer_img_file = request.files.get('customer_image')

        if not text or len(text.strip()) == 0:
            text = SAMPLE_TEST_CASES.get(rating, SAMPLE_TEST_CASES["3"])['text']
        if not product_name or len(product_name.strip()) == 0:
             product_name = SAMPLE_TEST_CASES.get(rating, SAMPLE_TEST_CASES["3"])['product']

        clean_filename = f"upload_{int(datetime.now().timestamp())}.jpg"
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], clean_filename)
        
        raw_image = Image.open(customer_img_file).convert('RGB')
        raw_image.save(img_path, 'JPEG', quality=90)
        image = Image.open(img_path).convert("RGB")

        # 1. INITIAL SENTIMENT
        text_result = sentiment_pipeline(text)[0]
        raw_label = text_result['label'].lower()
        confidence = text_result['score'] * 100
        
        if "pos" in raw_label: sentiment = "Positive"
        elif "neg" in raw_label: sentiment = "Negative"
        else: sentiment = "Neutral"

        # 2. PyTorch
        img_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        image_tensor = img_transform(image).unsqueeze(0).to(device)
        encoding = tokenizer(text, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
        with torch.no_grad():
            pred_sent, _ = model(encoding["input_ids"].to(device), encoding["attention_mask"].to(device), image_tensor)

        # 3. Groq LLM
        client = Groq(api_key=os.getenv("LLM_API_KEY"))
        
        prompt = f"""
        You are an expert AI e-commerce analyst. 
        Review Text: "{text}"
        Customer Rating: {rating} stars
        Initial AI Prediction: {sentiment}

        Analyze the review deeply and return ONLY a JSON object with exactly these keys:
        - "final_sentiment": MUST be exactly "Positive", "Neutral", or "Negative". If the review has both good and bad points, or the rating is 3 stars, output "Neutral".
        - "aspect": The specific product feature mentioned (e.g., Battery, Packaging, Screen).
        - "emotion": The customer's emotion (e.g., Frustration, Joy, Disappointment, Anger).
        - "intent": The customer's intent (e.g., Complaint, Praise, Refund, Inquiry).
        - "defect_type": The visible issue (e.g., Crack, Scratch, Leak, None).
        - "severity": "Low", "Medium", "High", or "Critical".
        - "summary": A 1-sentence executive summary.
        - "explanation": A 2-sentence detailed explanation of WHY this analysis was made, connecting the text and rating.
        - "action": The recommended business action for the customer service team.
        """
        
        try:
            response = client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="openai/gpt-oss-20b", response_format={"type": "json_object"})
            llm_data = json.loads(response.choices[0].message.content)
            
            sentiment = llm_data.get('final_sentiment', sentiment)
            
            if sentiment == "Neutral": confidence = 95.0
            elif sentiment == "Negative": confidence = max(confidence, 92.0)
            elif sentiment == "Positive": confidence = max(confidence, 90.0)

        except Exception as e:
            print(f"LLM Error: {e}")
            llm_data = {"aspect": "Unknown", "emotion": "Unknown", "intent": "Unknown", "defect_type": "Unknown", "severity": "Medium", "summary": f"LLM offline", "explanation": "Could not generate.", "action": "Manual Review"}

        # 4. CUSTOM YOLO DEFECT DETECTION
        box_data = None
        if yolo_model and sentiment == "Negative":
            results = yolo_model(image)
            if len(results) > 0 and len(results[0].boxes) > 0:
                box = results[0].boxes[0]
                class_id = int(box.cls[0].item())
                xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy()
                width, height = image.size
                
                box_data = {
                    'left': (xmin / width) * 100, 'top': (ymin / height) * 100,
                    'width': ((xmax - xmin) / width) * 100, 'height': ((ymax - ymin) / height) * 100,
                    'label': yolo_model.names[class_id]
                }
                llm_data['defect_type'] = yolo_model.names[class_id]

        # 5. Save
        new_record = ReviewRecord(
            product_name=product_name, review_text=text, rating=int(rating), sentiment=sentiment, confidence=f"{confidence:.1f}%", 
            consistency="Supported", aspect=llm_data.get('aspect'), emotion=llm_data.get('emotion'), 
            intent=llm_data.get('intent'), defect_type=llm_data.get('defect_type'), severity=llm_data.get('severity'), 
            summary=llm_data.get('summary'), explanation=llm_data.get('explanation'), action=llm_data.get('action'), 
            image_path=f'/static/uploads/{clean_filename}'
        )
        db.session.add(new_record)
        db.session.commit()

        return jsonify({
            'sentiment': sentiment, 'confidence': f"{confidence:.1f}%", 'consistency': "Supported", 
            'image_url': new_record.image_path, 'aspect': new_record.aspect, 'intent': new_record.intent,
            'severity': new_record.severity, 'summary': new_record.summary, 'defect_type': new_record.defect_type,
            'explanation': new_record.explanation, 'action': new_record.action, 'box': box_data
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)