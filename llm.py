import openai
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load API Key from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Create Flask app instance
app = Flask(__name__)
CORS(app)

@app.route('/chat', methods=['POST'])
def chat_with_llm():
    try:
        # Get user input from request
        user_input = request.json.get('input')
        
        # Call OpenAI API with user input
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=user_input,
            max_tokens=150
        )
        
        # Extract the response text
        response_text = response.choices[0].text.strip()
        
        # Return the response as JSON
        return jsonify({"response": response_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)