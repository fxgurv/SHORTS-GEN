




git clone https://github.com/fxgurv/mp.git
cd mp
git checkout update

# Create a virtual environment
python -m venv venv

# Activate the virtual environment - Windows
.\venv\Scripts\activate

# Activate the virtual environment - Unix
source venv/bin/activate


pip install -r requirements.txt


python src/main.py
