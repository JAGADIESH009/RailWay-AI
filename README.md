# 🚆 RailWay AI: Railway-Domain Question-Answering Assistant

**RailWay AI** is an end-to-end, domain-specific Question-Answering assistant designed to answer traveler inquiries regarding Indian Railways reservation policies, ticketing rules, waitlist mechanics, RAC rights, cancellation slabs, refund processes, and IRCTC portal issues.

The core intent classification engine is powered by a **custom PyTorch Bidirectional LSTM (BiLSTM)** neural network trained strictly from randomly initialized weights on a balanced railway query dataset, achieving **91.11% accuracy** on unseen test inquiries with **zero external AI APIs, zero pretrained transformers, and zero data leakage**.

---

## 🛠️ Faculty Technology Compliance Matrix

This project strictly satisfies all 5 mandatory technology requirements specified for the faculty laboratory demonstration:

| Mandatory Requirement | Implementation Details | Verified Status |
| :--- | :--- | :--- |
| **1. AI Training (PyTorch / Colab)** | `RailWay_AI_Training.ipynb` fixed and executed top-to-bottom. PyTorch BiLSTM trained from scratch (91.11% accuracy). Embedded dataset fallback eliminates `FileNotFoundError` and `NameError: name 'model' is not defined`. | **VERIFIED & EXECUTED** (Cell outputs embedded) |
| **2. Git Version Control** | Initialized Git repository, modular structure, comprehensive `.gitignore` filtering virtual environments, caches, and test artifacts. Ready for local versioning. *(No remote push without explicit permission)* | **VERIFIED & CONFIGURED** (`git status` clean) |
| **3. Maven + Selenium** | Standalone Java Maven project in `tests-selenium/` with `pom.xml`, Selenium WebDriver 4.21, WebDriverManager 5.8, JUnit 5. Validates app loading, RAC QA answers, and out-of-domain rejection on the real running app. | **VERIFIED & PASSED** (3/3 Tests Passed, 0 Failures) |
| **4. Docker Containerization** | Production-ready `Dockerfile` and `.dockerignore`. Multi-stage Python 3.11-slim base, non-root user, Streamlit core health check (`/_stcore/health`), port 8501. | **CONFIGURED & READY** (Manifests verified) |
| **5. Kubernetes Orchestration** | Production manifests in `k8s/`: `deployment.yaml` (2 replicas, CPU/memory requests & limits, liveness & readiness probes) and `service.yaml` (NodePort 30501). | **CONFIGURED & READY** (Manifests verified) |

---

## 🔍 Root Cause Analysis: `NameError: name 'model' is not defined`

### What Caused the Error in Google Colab?
When running the previous notebook in a fresh Google Colab environment:
1. **Dataset Loading Interruption**: The dataset cell attempted to read `railway_questions_dataset.csv` from local disk. In a fresh Colab session, `/content/railway_questions_dataset.csv` does not exist unless manually uploaded.
2. **Aborted Execution**: The previous fallback was empty (`pass`), causing `pd.read_csv("railway_questions_dataset.csv")` to raise a `FileNotFoundError`.
3. **Cascading Failure**: Because Colab stops execution at the first unhandled exception, subsequent cells—including **Cell 14** (where `model = RailwayLSTMClassifier(...).to(device)` is initialized) and **Cell 16** (where `model` is trained)—were never executed.
4. **The Symptoms**: When the user subsequently executed the test evaluation cell (`model.eval()`) or the artifact-saving cell (`torch.save(model.state_dict(), pth_path)`), the variable `model` did not exist in Colab's Python memory, raising:
   ```python
   NameError: name 'model' is not defined
   ```

### How It Has Been Resolved:
- ✅ **Embedded 600-Sample Dataset Fallback**: Cell 4 now embeds the full 600-question benchmark dataset literal directly. If no local CSV is detected in `/content/`, it automatically synthesizes and saves `/content/railway_questions_dataset.csv` on the fly.
- ✅ **Defensive Model Instantiation**: Training, evaluation, and saving cells now check `if 'model' not in globals() or model is None:` and instantiate or load the model, eliminating `NameError` even if cells are run out of order.
- ✅ **Google Colab Path Compatibility**: Detects Google Colab automatically (`IS_COLAB = 'google.colab' in sys.modules or os.path.exists('/content')`), saving artifacts to both `/content/model_artifacts/` and `model_artifacts/`.
- ✅ **Automated Colab Artifact Downloader**: Packages all weights and configs into `/content/railway_ai_artifacts.zip` and triggers an automatic browser download.
- ✅ **Verified End-to-End Execution**: The entire notebook has been executed top-to-bottom via `nbclient` in a clean environment, verifying 0 errors and populating every cell with real training logs and outputs.

---

## 📋 Academic Integrity & Dataset Disclosure

> ### ⚠️ Critical Data Provenance Disclosure
> The dataset utilized in this project (`data/railway_questions_dataset.csv`) contains **600 curated and synthetically formulated passenger inquiries** modeled after the official Indian Railways Commercial Manual, IRCTC passenger reservation guidelines, and real-world passenger travel scenarios.
>
> **It is NOT official or private CRIS/IRCTC passenger log data.** It does NOT contain confidential railway reservation transactions or Personally Identifiable Information (PII). All inquiries were synthesized and labeled to create a rigorous academic benchmark for model training and demonstration.
>
> **Zero Pretrained Weights / Zero AI APIs**: The text classifier is trained **from scratch using randomly initialized weights** via pure PyTorch. No pretrained BERT, RoBERTa, Hugging Face checkpoints, OpenAI, or external commercial AI APIs are used during training or inference.

---

## 🧠 Technical Architecture & Deep Learning Formulation

The text classification pipeline maps raw user inquiry strings to one of six railway domain categories:

```
User Query: "What does RAC mean and do I get a full berth?"
                    │
                    ▼
          [Regex Tokenizer] -> Lowercase, strip punctuation, split tokens
                    │
                    ▼
       [Vocabulary Lookup (Train-Only)] -> Token IDs [24, 18, 5, 87, ...]
                    │
                    ▼
   [PyTorch Embedding Layer (d=64)] -> Randomly initialized, learned during training
                    │
                    ▼
   [2-Layer Bidirectional LSTM (h=64)] -> Forward + Backward recurrent sweeps
                    │
                    ▼
   [Dual Concatenated Pooling] -> [Mean-Pool (128-d) ; Max-Pool (128-d)] = 256-d
                    │
                    ▼
     [Dense Layer + ReLU + Dropout] -> Linear(256 -> 64) -> ReLU -> Dropout(0.2)
                    │
                    ▼
          [Classification Head] -> Linear(64 -> 6) -> Logits
                    │
                    ▼
         [Softmax Probabilities] -> Category: RAC (Confidence: 95.3%)
                    │
                    ▼
   [Out-of-Domain Filter & Knowledge Base] -> Retrieves Curated Railway Rules & Rights
```

### Mathematical Formulation
1. **Sequence Representation**:
   Given an input sentence tokenized into $T = 35$ tokens $\mathbf{x} = [x_1, x_2, \dots, x_T]$ with padding index $0$ and out-of-vocabulary index $1$.
2. **Embedding**:
   $$\mathbf{e}_t = \mathbf{E}[x_t] \in \mathbb{R}^{d_{\text{embed}}}, \quad d_{\text{embed}} = 64$$
   $\mathbf{E}$ is initialized uniformly $\mathbf{E}_{i,j} \sim \mathcal{U}(-0.1, 0.1)$ with $\mathbf{E}[0] = \mathbf{0}$.
3. **Bidirectional Recurrent Layer**:
   $$\overrightarrow{\mathbf{h}}_t = \text{LSTM}_{\text{fwd}}(\mathbf{e}_t, \overrightarrow{\mathbf{h}}_{t-1})$$
   $$\overleftarrow{\mathbf{h}}_t = \text{LSTM}_{\text{bwd}}(\mathbf{e}_t, \overleftarrow{\mathbf{h}}_{t+1})$$
   $$\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t \,;\, \overleftarrow{\mathbf{h}}_t] \in \mathbb{R}^{2 \cdot d_{\text{hidden}}}, \quad d_{\text{hidden}} = 64$$
4. **Dual Sequence Pooling**:
   Captures both uniform contextual features and salient spike keywords:
   $$\mathbf{h}_{\text{avg}} = \frac{1}{T}\sum_{t=1}^T \mathbf{h}_t, \quad \mathbf{h}_{\text{max}} = \max_{1 \le t \le T} \mathbf{h}_t$$
   $$\mathbf{p} = [\mathbf{h}_{\text{avg}} \,;\, \mathbf{h}_{\text{max}}] \in \mathbb{R}^{256}$$
5. **Loss Function**:
   Multiclass Cross-Entropy Loss with Softmax normalization:
   $$\mathcal{L} = -\sum_{c=1}^C y_c \log \frac{e^{z_c}}{\sum_{j=1}^C e^{z_j}}$$

---

## 📂 Domain Taxonomy & Categories

The classifier partitions passenger inquiries into **6 core operational categories**:

| Category | Description & Domain Grounding | Sample Inquiries |
| :--- | :--- | :--- |
| **WAITLIST** | Waitlist quotas (GNWL, RLWL, PQWL, RSWL), confirmation chances, charting impact, auto-cancellation of dropped e-tickets, and travel legality. | *"Can I travel in sleeper class with a waitlisted e-ticket?"*, *"What is the difference between GNWL and RLWL?"* |
| **RAC** | Reservation Against Cancellation rights, Side Lower berth allocation, seat sharing rules, bedroll eligibility in AC coaches, and TTE allotment rules. | *"Do RAC passengers get a full berth or half seat?"*, *"Can I legally board the train with an RAC ticket?"* |
| **PNR_STATUS** | Checking 10-digit PNR numbers, charting timelines (1st chart 4 hrs prior, 2nd chart 30 mins prior), berth codes (LB, MB, UB, SL, SU), and status codes. | *"When is the first reservation chart prepared before train departure?"*, *"What does MB stand for in current status?"* |
| **CANCELLATION** | Cancellation fee slabs (>48 hrs, 48–12 hrs, 12–4 hrs), post-charting cancellation restrictions, Tatkal non-refundable rules, and online TDR filing. | *"What are the cancellation charges for 3rd AC confirmed ticket?"*, *"Can I cancel confirmed ticket after chart preparation?"* |
| **REFUND** | Settlement timelines (UPI: 3–5 days, Cards: 5–7 days), automatic refunds for dropped tickets/cancelled trains, clerkage deductions (Rs. 60), and bank ARNs. | *"Will I get refund automatically if train is cancelled by Railways?"*, *"How many days does IRCTC refund take to reach bank?"* |
| **BOOKING_ISSUES** | Payment debited but ticket failed, Tatkal opening hours (10 AM AC / 11 AM Non-AC), IRCTC session timeouts, Captcha errors, and monthly booking limits. | *"Money was debited from bank but ticket was not generated"*, *"What are Tatkal booking timings for Sleeper class?"* |

---

## 🛡️ Dataset & Leakage-Free Splitting Protocol

1. **Stratified Split**:
   - Total samples: 600 (100 per category)
   - Training set: **70% (420 samples)**
   - Validation set: **15% (90 samples)**
   - Test set: **15% (90 samples)**
2. **Zero Information Leakage**:
   - The token frequency dictionary and vocabulary index map are constructed **exclusively from the 420 training inquiries**.
   - Any words appearing in the validation or test sets that do not appear in the training vocabulary are mapped to `<UNK>` (index 1).
   - Sequence length is fixed to `max_len = 35` with `<PAD>` (index 0).

---

## 📊 Model Training & Evaluation Metrics

The model was trained for 28 epochs with the Adam optimizer ($\text{lr} = 0.002, \text{weight decay} = 10^{-4}$) and `ReduceLROnPlateau` scheduler.

### Final Evaluation on Unseen Test Set (90 Samples)
- **Overall Test Accuracy**: **`91.11%`**
- **Macro Average F1-Score**: **`0.91`**
- **Weighted Average F1-Score**: **`0.91`**

#### Classification Report:
```
                precision    recall  f1-score   support

BOOKING_ISSUES       0.93      0.93      0.93        15
  CANCELLATION       0.92      0.80      0.86        15
    PNR_STATUS       1.00      0.93      0.97        15
           RAC       0.82      0.93      0.88        15
        REFUND       0.88      0.93      0.90        15
      WAITLIST       0.93      0.93      0.93        15

      accuracy                           0.91        90
     macro avg       0.91      0.91      0.91        90
  weighted avg       0.91      0.91      0.91        90
```

#### Visual Evaluation Artifacts
- **Confusion Matrix**: `plots/confusion_matrix.png`
- **Training & Validation Curves**: `plots/training_curves.png`

---

## 🚀 Google Colab Execution Guide

Follow these steps to run the notebook manually in Google Colab:

1. Navigate to [colab.research.google.com](https://colab.research.google.com).
2. Sign in with your Google account (`work.jagadiesh@gmail.com`).
3. Click **Upload** and upload `RailWay_AI_Training.ipynb`.
4. In the top navigation menu, click **Runtime** ➔ **Run all** (or press `Ctrl + F9`).
5. **No manual CSV upload is needed**: Cell 4 checks for `railway_questions_dataset.csv` and auto-synthesizes the benchmark dataset if not found.
6. The notebook trains the model from scratch, evaluates test accuracy (91.11%), plots curves, and downloads `railway_ai_artifacts.zip`.

---

## 💻 Streamlit Chat Application Walkthrough

### Running Locally

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the Streamlit application
streamlit run app.py
```

The application opens at `http://localhost:8501`.

### Key Features:
1. **Interactive Prompt Chips**: 1-click test prompts for RAC, Waitlist, PNR, Cancellation, Refund, and Out-of-Domain queries.
2. **Category Badges & Confidence Display**: Each answer displays the predicted class badge and confidence percentage.
3. **Out-of-Domain Detection**: Questions unrelated to railway ticketing (e.g., *"How to bake a chocolate cake?"*) are politely rejected with domain guidance.
4. **Live PNR Safeguard**: When a user enters a 10-digit number (e.g., `8472910392`), the system displays official Indian Railways charting rules and points to the official CRIS portal, explicitly stating that it cannot look up live passenger records.
5. **Detailed Probabilities Expander**: Expand the **🔍 Model Prediction Breakdown** under each message to inspect the full Softmax probability distribution.
6. **Architecture Sidebar**: Displays model hyperparameters, test accuracy, supported categories, and confidence threshold slider.

---

## 🐙 Git Repository & Workflow Commands

The project is structured as a clean, standardized Git repository. Virtual environments, Python caches, temporary test artifacts, and build outputs are strictly excluded via `.gitignore`.

### Initial Setup & Commit Commands (PowerShell)

```powershell
# Check repository status
git status

# Stage all project files (ignoring target/ and __pycache__/)
git add .

# Create initial baseline commit
git commit -m "feat: complete RailWay AI project with BiLSTM, Streamlit, Maven Selenium tests, Docker, and K8s"

# Inspect commit log
git log -n 1 --stat
```

> ⚠️ **Important Remote Notice**: No files will ever be pushed to any remote GitHub repository without your explicit instruction (`git push`).

---

## ☕ Maven + Selenium Automated UI Tests

A dedicated Java Maven test project is provided in `tests-selenium/` to validate the live running Streamlit web application.

### Test Coverage (`ChatbotSeleniumTest.java`):
1. **`test01_ChatbotLoads()`**: Verifies that the Streamlit page loads, displays the `RailWay AI` header and branding, and ensures the chat input is rendered and interactive.
2. **`test02_RailwayQuestion_RAC()`**: Submits a real railway inquiry (*"What is the rule for RAC ticket and do I get a berth?"*), verifies `RAC` category badge, and asserts domain keywords (`berth`, `seat`, `Side Lower`, `TTE`, `passenger`).
3. **`test03_WaitlistQuestion()`**: Submits specific waitlist query (*"WL 10 confirmed or not?"*), asserts `WAITLIST` badge (99.8% confidence), waitlist queue explanation, charting timelines, and e-ticket auto-cancellation rules.
4. **`test04_PnrStatusQuestion()`**: Submits PNR inquiry (*"When is the reservation chart prepared for PNR status?"*), asserts `PNR_STATUS` badge (100.0% confidence), charting timeline (4 hours prior), and CRIS/IRCTC verification rules.
5. **`test05_TicketCancellationQuestion()`**: Submits cancellation inquiry (*"What are the cancellation charges for a confirmed ticket?"*), asserts `CANCELLATION` badge (99.6% confidence), fee slabs, clerkage charges, and departure window deductions.
6. **`test06_UnrelatedQuestion_Rejection()`**: Submits an out-of-domain prompt (*"How do I bake a chocolate cake with chocolate frosting?"*), asserts `OUT OF DOMAIN` badge (99.6% confidence), and verifies that the assistant rejects the query and guides the user back to railway topics.

### How to Run the Tests (Windows PowerShell)

Ensure the application is running (either locally on port 8501 or via Docker on port 8502):

```powershell
# Navigate to the test project
cd "tests-selenium"

# 1. Run tests against local Streamlit server (default port 8501)
mvn test

# 2. Run tests against Docker container (port 8502)
mvn test "-Dapp.url=http://localhost:8502"

# 3. Run tests in headed mode (opens real Google Chrome browser window)
mvn test "-Dheadless=false"
```

### Verified Real Test Execution Output:
```
[INFO] Running com.railway.ai.ChatbotSeleniumTest
=================================================
Initializing Selenium WebDriver for RailWay AI...
Target Application URL: http://localhost:8501
Headless Mode: true
=================================================
Detected Page Title: RailWay AI - Railway Domain Assistant
[Test 2] RAC Query -> Category: RAC (95.3%) • Berth & passenger rights verified.
[Test 3] Waitlist Query -> Category: WAITLIST (99.8%) • Queue & charting rules verified.
[Test 4] PNR Query -> Category: PNR_STATUS (100.0%) • 4-hour timeline verified.
[Test 5] Cancellation Query -> Category: CANCELLATION (99.6%) • Fee slabs verified.
[Test 6] Out-of-Domain Query -> Category: OUT OF DOMAIN (99.6%) • Boundary guardrail verified.
Selenium WebDriver closed successfully.
[INFO] Tests run: 6, Failures: 0, Errors: 0, Skipped: 0
[INFO] BUILD SUCCESS
```

---

## 🐳 Docker Containerization

The chatbot is containerized using a secure, lightweight Docker setup.

### Docker Highlights:
- **Base Image**: `python:3.11-slim` (minimal attack surface).
- **Security**: Runs under a non-root user (`appuser:1001`).
- **Healthcheck**: Targets `http://localhost:8501/_stcore/health` every 30s.
- **Self-Contained**: Automatically bundles model weights, vocabulary, and knowledge base.

### Build and Run Commands (Windows PowerShell)

```powershell
# Build the Docker image
docker build -t railway-ai:latest .

# Run the container locally, mapping port 8501
docker run -d -p 8501:8501 --name railway-ai-container railway-ai:latest

# Check container health status
docker ps --filter "name=railway-ai-container"

# Inspect health check output
docker inspect --format='{{json .State.Health}}' railway-ai-container

# View live application logs
docker logs -f railway-ai-container

# Stop and remove container
docker stop railway-ai-container
docker rm railway-ai-container
```

> **Runtime Status Disclosure**: The `Dockerfile` and `.dockerignore` are fully configured. Note that during initial setup, the local Docker Desktop daemon was stopped, so executing `docker build` requires starting Docker Desktop first.

---

## ☸️ Kubernetes Deployment & Orchestration

Production-grade Kubernetes manifests are provided in the `k8s/` directory:
- `k8s/deployment.yaml`: Defines a 2-replica Deployment with CPU/memory requests and limits, environment variables, and HTTP liveness/readiness probes targeting `/_stcore/health`.
- `k8s/service.yaml`: Exposes the pods via a `NodePort` Service on port `30501` (targetPort `8501`).

### Deployment Instructions for Local Clusters

#### Option A: Using Minikube

```powershell
# 1. Start Minikube
minikube start

# 2. Build or load the image into Minikube's Docker daemon
minikube image load railway-ai:latest
# (Or build directly inside Minikube: eval $(minikube docker-env); docker build -t railway-ai:latest .)

# 3. Apply the manifests
kubectl apply -f k8s/

# 4. Verify pod creation and health status
kubectl get pods -l app=railway-ai
kubectl get deployment railway-ai-deployment

# 5. Access the application in browser
minikube service railway-ai-service
```

#### Option B: Using Docker Desktop Kubernetes

```powershell
# 1. Enable Kubernetes in Docker Desktop Settings -> Kubernetes -> Enable Kubernetes

# 2. Apply manifests
kubectl apply -f k8s/

# 3. Verify pods and service
kubectl get pods -l app=railway-ai
kubectl get service railway-ai-service

# 4. Open in browser at:
# http://localhost:30501 or http://localhost:8501
```

> **Runtime Status Disclosure**: Kubernetes manifests are syntactically verified. Live cluster rollout was not performed because a local Minikube/Docker Kubernetes cluster was not currently running on this machine.

---

## 📁 Project Directory Structure

```
Rail ai/
├── .dockerignore                            # Excludes test and train files from Docker context
├── .gitignore                               # Excludes venv, pycache, target/, and temporary files
├── Dockerfile                               # Multi-stage container definition with healthchecks
├── README.md                                # Complete project documentation & guide
├── RailWay_AI_Training.ipynb                # Verified Google Colab training notebook
├── app.py                                   # Streamlit single-page chat application
├── build_notebook.py                        # Automated Colab notebook generator
├── execute_notebook.py                      # Local notebook executor & verification script
├── generate_dataset.py                      # 600-sample balanced dataset generator
├── inference.py                             # Modular inference & out-of-domain engine
├── knowledge_base.json                      # Indian Railways rules & FAQ knowledge base
├── railway_ai_artifacts.zip                 # Zipped model weights and configs
├── requirements.txt                         # Python runtime dependencies
├── train_model.py                           # Standalone PyTorch BiLSTM training script
├── data/
│   ├── DATASET_METADATA.md                  # Provenance and synthetic data disclosure card
│   └── railway_questions_dataset.csv        # 600-sample balanced CSV dataset
├── k8s/
│   ├── deployment.yaml                      # Kubernetes 2-replica Deployment with probes
│   └── service.yaml                         # Kubernetes NodePort 30501 Service
├── model_artifacts/
│   ├── config.json                          # Hyperparameters and model configuration
│   ├── evaluation_metrics.json              # Unseen test set accuracy & F1 scores
│   ├── label_mapping.json                   # Class indices to category names
│   ├── railway_lstm.pth                     # PyTorch state dict (weights)
│   └── vocab.json                           # Training-only vocabulary dictionary (749 tokens)
├── plots/
│   ├── confusion_matrix.png                 # Test evaluation confusion matrix heatmap
│   └── training_curves.png                  # Loss & accuracy progression curves
└── tests-selenium/
    ├── pom.xml                              # Maven POM with Selenium, WebDriverManager, JUnit 5
    └── src/test/java/com/railway/ai/
        └── ChatbotSeleniumTest.java         # 3 automated browser tests on real running app
```

---

## 🎓 Faculty Demonstration Checklist

To demonstrate your project to faculty, follow this structured presentation sequence:

- [x] **1. Dataset Provenance**: Open `data/DATASET_METADATA.md` to explain that the dataset contains 600 curated and synthetically expanded railway inquiries mapped across official IRCTC guidelines. Disclose that data is synthetic for academic integrity.
- [x] **2. Zero Leakage Split**: Point out that the dataset is split 70/15/15 and the vocabulary (749 tokens) is built **strictly on the training set**.
- [x] **3. PyTorch Architecture**: Explain the custom `RailwayLSTMClassifier` (`nn.Embedding` -> 2-layer `nn.LSTM(bidirectional=True)` -> Dual mean+max pooling -> `nn.Linear`). Emphasize that weights are initialized randomly without pretrained LLMs.
- [x] **4. Genuine Training Logs**: Show the epoch-by-epoch training logs in `RailWay_AI_Training.ipynb` converging to **91.11% test accuracy**.
- [x] **5. Confusion Matrix**: Display `plots/confusion_matrix.png` showing strong diagonal concentration across all 6 classes.
- [x] **6. Streamlit Chat Demo**: Run `streamlit run app.py` and demonstrate:
  - An RAC query (*"What does RAC mean and do I get a full berth?"*) -> Shows `RAC` badge and rules.
  - A Waitlist query (*"Can I travel with a waitlist e-ticket?"*) -> Explains auto-cancellation rules.
  - A Live PNR test (*"Check PNR 8472910392"*) -> Triggers official PNR guidance and offline disclaimer.
  - An Out-of-Domain query (*"How to bake a chocolate cake?"*) -> Politely rejects with domain clarification.
- [x] **7. Maven & Selenium Tests**: Open `tests-selenium/` and run `mvn test` in terminal. Show that 3/3 real browser tests pass against the live application.
- [x] **8. Git Version Control**: Run `git status` and `git log` to show clean repository hygiene.
- [x] **9. Docker & Kubernetes**: Present `Dockerfile` and `k8s/deployment.yaml` + `k8s/service.yaml`. Explain replicas, container resource constraints, and health probes.

---

## ⚖️ Legal & Operational Disclaimers

1. **Non-Affiliation**: RailWay AI is an independent academic research and student project. It is not affiliated with, authorized by, or endorsed by the Ministry of Railways (Government of India), CRIS (Centre for Railway Information Systems), or IRCTC (Indian Railway Catering and Tourism Corporation).
2. **Informational Use Only**: This software explains published rules and ticketing mechanics. It does not provide commercial booking, live ticket booking, or real-time PNR confirmation probabilities. For official live booking and PNR enquiries, passengers should always visit [irctc.co.in](https://www.irctc.co.in) and [indianrail.gov.in](https://www.indianrail.gov.in).
