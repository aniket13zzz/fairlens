# 1. Clone / unzip the project
cd fairlens-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Configure API keys for AI explanations
cp .env.example .env
# Edit .env — add ANTHROPIC_API_KEY or OPENAI_API_KEY

# 4. Run
python run.py
