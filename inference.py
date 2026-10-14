"""
Inference Engine for RailWay AI
Loads trained PyTorch Bidirectional LSTM model, vocabulary, and knowledge base.
Provides question classification, confidence estimation, out-of-domain filtering,
and curated railway answer retrieval.
"""

import os
import re
import json
import torch
import torch.nn as nn

class RailwayLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=64, num_classes=6, 
                 num_layers=2, bidirectional=True, dropout=0.3, pad_idx=0):
        super(RailwayLSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.dropout1 = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(lstm_out_dim * 2, 64)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        avg_pool = torch.mean(lstm_out, dim=1)
        max_pool, _ = torch.max(lstm_out, dim=1)
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        
        x = self.dropout1(pooled)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        logits = self.fc2(x)
        return logits

class RailwayAssistant:
    def __init__(self, artifacts_dir="model_artifacts", kb_path="knowledge_base.json"):
        self.artifacts_dir = artifacts_dir
        self.kb_path = kb_path
        self.device = torch.device("cpu")
        self.load_resources()
        
    def load_resources(self):
        # 1. Config
        config_file = os.path.join(self.artifacts_dir, "config.json")
        with open(config_file, "r", encoding="utf-8") as f:
            self.config = json.load(f)
            
        # 2. Vocabulary
        vocab_file = os.path.join(self.artifacts_dir, "vocab.json")
        with open(vocab_file, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
            
        # 3. Label Mapping
        label_file = os.path.join(self.artifacts_dir, "label_mapping.json")
        with open(label_file, "r", encoding="utf-8") as f:
            label_data = json.load(f)
            self.classes = label_data["classes"]
            self.label2idx = label_data["label2idx"]
            self.idx2label = {int(k): v for k, v in label_data["idx2label"].items()}
            
        # 4. Model
        self.model = RailwayLSTMClassifier(
            vocab_size=self.config["vocab_size"],
            embedding_dim=self.config["embedding_dim"],
            hidden_dim=self.config["hidden_dim"],
            num_classes=self.config["num_classes"],
            num_layers=self.config["num_layers"],
            bidirectional=self.config["bidirectional"],
            dropout=self.config["dropout"],
            pad_idx=self.config["pad_idx"]
        ).to(self.device)
        
        weights_file = os.path.join(self.artifacts_dir, "railway_lstm.pth")
        state_dict = torch.load(weights_file, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # 5. Knowledge Base
        with open(self.kb_path, "r", encoding="utf-8") as f:
            self.kb = json.load(f)
            
    def tokenize(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return text.strip().split()
        
    def text_to_sequence(self, tokens, max_len=35):
        unk_idx = self.config.get("unk_idx", 1)
        pad_idx = self.config.get("pad_idx", 0)
        seq = [self.vocab.get(t, unk_idx) for t in tokens]
        if len(seq) < max_len:
            seq = seq + [pad_idx] * (max_len - len(seq))
        else:
            seq = seq[:max_len]
        return seq

    def check_out_of_domain_heuristics(self, query, tokens):
        # Known railway keywords
        railway_keywords = {
            "pnr", "train", "railway", "ticket", "tickets", "berth", "seat", "seats", "coach", "tatkal",
            "rac", "waitlist", "waiting", "wl", "gnwl", "rlwl", "pqwl", "rswl", "ckwl", "ldwl", "chart",
            "charting", "cancel", "cancelled", "cancellation", "cancelling", "refund", "refunds", "refunded",
            "irctc", "tte", "sleeper", "ac", "3ac", "2ac", "1ac", "3e", "2s", "cc", "ec", "vande",
            "bharat", "rajdhani", "shatabdi", "duronto", "garib", "rath", "tdr", "station", "quota",
            "clerkage", "boarding", "bedroll", "blanket", "passenger", "passengers", "booking", "book",
            "booked", "fail", "failed", "debited", "deducted", "aadhaar", "otp", "captcha", "fare",
            "payment", "upgrade", "upgradation", "counter", "prs", "concession", "senior", "citizen",
            "infant", "child", "journey", "diverted", "delayed", "delay", "platform"
        }
        
        stopwords = {
            "what", "is", "the", "in", "a", "an", "to", "for", "of", "and", "or", "how", "do", "does",
            "can", "i", "my", "why", "who", "where", "which", "when", "on", "at", "by", "if", "not",
            "no", "yes", "me", "you", "be", "with", "from", "there", "about", "tell", "give", "please",
            "will", "get", "got", "much", "many", "some", "any", "after", "before", "between"
        }
        
        # Explicit non-railway indicators (flights, buses, movies, cooking, etc.)
        non_railway_indicators = {
            "flight", "flights", "airplane", "airline", "airlines", "indigo", "airindia",
            "spicejet", "airport", "bus", "buses", "redbus", "movie", "movies", "cinema",
            "hotel", "hotels", "weather", "recipe", "cake", "pizza", "burger", "cooking",
            "president", "prime minister", "capital", "currency", "crypto", "bitcoin",
            "football", "cricket", "ipl", "fifa", "song", "lyrics", "poem"
        }
        has_non_railway_topic = any(kw in query.lower() for kw in non_railway_indicators)

        # Count overlapping domain keywords
        overlap = [t for t in tokens if t in railway_keywords]
        
        # Meaningful non-stopwords
        content_tokens = [t for t in tokens if t not in stopwords]
        
        # Check percentage of known vocab tokens among content tokens
        known_content_tokens = [t for t in content_tokens if t in self.vocab and self.vocab[t] > 1]
        content_known_ratio = len(known_content_tokens) / max(len(content_tokens), 1)
        
        # Look for 10-digit number (PNR)
        has_10_digit_pnr = bool(re.search(r"\b\d{10}\b", query))
        
        return {
            "has_10_digit_pnr": has_10_digit_pnr,
            "has_non_railway_topic": has_non_railway_topic,
            "railway_keyword_count": len(overlap),
            "content_token_count": len(content_tokens),
            "content_known_ratio": content_known_ratio
        }

    def predict(self, query, confidence_threshold=0.50):
        """
        Classifies the user query, retrieves knowledge base answers,
        and flags out-of-domain / live status requests.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return {
                "category": "EMPTY",
                "confidence": 0.0,
                "is_out_of_domain": True,
                "response_text": "Please enter a railway-related question (e.g., questions regarding Waitlist, RAC, PNR status, Cancellation charges, Refunds, or IRCTC Booking issues)."
            }
            
        tokens = self.tokenize(cleaned_query)
        heuristics = self.check_out_of_domain_heuristics(cleaned_query, tokens)
        
        # Specific Handler: User input looks like a live PNR number query
        if heuristics["has_10_digit_pnr"]:
            pnr_match = re.search(r"\b\d{10}\b", cleaned_query).group(0)
            return {
                "category": "PNR_STATUS",
                "confidence": 0.99,
                "is_pnr_lookup_attempt": True,
                "is_out_of_domain": False,
                "display_title": f"PNR Enquiry Rules (PNR: {pnr_match})",
                "response_text": (
                    f"**Notice Regarding Live PNR Tracking:**\n\n"
                    f"RailWay AI is an educational domain-knowledge system. **It does not connect to live Indian Railways/CRIS databases and cannot look up real-time status or confirmation probability for PNR `{pnr_match}`.**\n\n"
                    f"### How to Check Official Live Status:\n"
                    f"- Visit the official Indian Railways Passenger Enquiry Portal: [indianrail.gov.in/enquiry/PNR](https://www.indianrail.gov.in/enquiry/PNR/PnrEnquiry.html)\n"
                    f"- Or SMS your 10-digit PNR to **139**.\n\n"
                    f"### Key Charting & PNR Rules to Know:\n"
                    f"- **1st Chart**: Prepared 4 hours before train departure. Coach and berth allocations are finalized then.\n"
                    f"- **2nd Chart**: Prepared 30–45 minutes prior to departure for any remaining current bookings."
                ),
                "key_points": self.kb["PNR_STATUS"]["key_points"],
                "official_portal": "https://www.indianrail.gov.in/enquiry/PNR/PnrEnquiry.html"
            }
            
        # Model forward pass
        seq = self.text_to_sequence(tokens, max_len=self.config.get("max_seq_len", 35))
        tensor_in = torch.tensor([seq], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            logits = self.model(tensor_in)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            top_prob, top_idx = torch.topk(probs, k=1)
            
        confidence = float(top_prob.item())
        pred_category = self.idx2label[top_idx.item()]
        
        # Check if query is unrelated / out-of-domain:
        # 1. Model confidence below threshold
        # 2. No railway domain keywords detected AND low content token match
        is_ood = (
            heuristics["has_non_railway_topic"] or
            confidence < confidence_threshold or 
            (heuristics["railway_keyword_count"] == 0 and heuristics["content_known_ratio"] < 0.40) or
            (heuristics["railway_keyword_count"] == 0 and heuristics["content_token_count"] >= 2)
        )
        
        if is_ood:
            return {
                "category": "OUT_OF_DOMAIN",
                "confidence": round(confidence, 4),
                "is_out_of_domain": True,
                "display_title": "Out of Railway Domain",
                "response_text": (
                    "I am **RailWay AI**, a specialized assistant trained exclusively on railway reservation policies, "
                    "ticketing procedures, waitlist rules, RAC rights, cancellation slabs, refunds, and IRCTC portal issues.\n\n"
                    "Your question does not appear to be related to Indian Railways. Please ask questions such as:\n"
                    "- *'What happens if my ticket is still on the waitlist after chart prep?'*\n"
                    "- *'Do RAC passengers get a full berth and bedroll?'*\n"
                    "- *'What are the cancellation charges for 3rd AC 24 hours before train departure?'*\n"
                    "- *'Money was deducted from my bank but no ticket was booked.'*"
                ),
                "all_probabilities": {self.idx2label[i]: round(float(probs[i]), 4) for i in range(len(self.classes))}
            }
            
        # Retrieve Curated Knowledge Base content
        kb_entry = self.kb.get(pred_category, {})
        title = kb_entry.get("title", pred_category)
        overview = kb_entry.get("overview", "")
        key_points = kb_entry.get("key_points", [])
        actionable_guidance = kb_entry.get("actionable_guidance", "")
        official_portal = kb_entry.get("official_portal", "https://www.irctc.co.in")
        
        # Find closest subtopic Q&A match within the category
        best_qa = None
        max_matches = 0
        for qa in kb_entry.get("frequently_asked", []):
            match_count = sum(1 for kw in qa.get("keywords", []) if kw in cleaned_query.lower())
            if match_count > max_matches:
                max_matches = match_count
                best_qa = qa
                
        # Format comprehensive response
        formatted_response = f"### {title}\n\n"
        if best_qa and max_matches > 0:
            formatted_response += f"**Direct Answer:**\n{best_qa['answer']}\n\n"
        formatted_response += f"**Overview:**\n{overview}\n\n"
        formatted_response += "**Official Railway Rules:**\n"
        for kp in key_points:
            formatted_response += f"- {kp}\n"
        if actionable_guidance:
            formatted_response += f"\n💡 **Actionable Advice:** {actionable_guidance}\n"
            
        return {
            "category": pred_category,
            "confidence": round(confidence, 4),
            "is_out_of_domain": False,
            "display_title": title,
            "response_text": formatted_response,
            "key_points": key_points,
            "actionable_guidance": actionable_guidance,
            "official_portal": official_portal,
            "all_probabilities": {self.idx2label[i]: round(float(probs[i]), 4) for i in range(len(self.classes))}
        }

if __name__ == "__main__":
    assistant = RailwayAssistant()
    test_queries = [
        "What does RAC mean and do I get a full berth?",
        "If my ticket is waitlisted will I get a full refund if not confirmed?",
        "Can I cancel my confirmed ticket after chart preparation?",
        "Check status for PNR 4529104829",
        "Money debited from bank but ticket was not generated",
        "Who is the president of France?"
    ]
    print("\n--- Running Inference Verification Tests ---\n")
    for q in test_queries:
        res = assistant.predict(q)
        print(f"Q: '{q}'")
        print(f"-> Category: {res['category']} | Confidence: {res['confidence']*100:.1f}% | OutOfDomain: {res['is_out_of_domain']}")
        print("-" * 50)
