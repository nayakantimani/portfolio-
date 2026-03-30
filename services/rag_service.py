# services/rag_service.py
import os
import json
import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

DB_PATH = "../data/knowledge_base_db"
COLLECTION_NAME = "therapeutic_knowledge_base"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = 'gemini-2.5-pro'

class RAG_LLM_System:
    def __init__(self):
        print("🤖 Initializing RAG+LLM System...")
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = self.client.get_collection(name=COLLECTION_NAME)
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu')
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found. Make sure it is set in your .env file.")
            
        genai.configure(api_key=api_key)
        self.llm_model = genai.GenerativeModel(LLM_MODEL_NAME)
        print("✅ RAG+LLM System is ready.")

    def _search_knowledge_base(self, query_text: str, n_results=5):
        """Searches the knowledge base for relevant context."""
        print(f"    -> Searching knowledge base with query: '{query_text}'")
        query_embedding = self.embedding_model.encode(query_text).tolist()
        results = self.collection.query(query_embeddings=[query_embedding], n_results=n_results)
        return results['documents'][0]

    def generate_clarifying_questions(self, original_text: str, emotion_data: dict):
        """
        Generates clarifying questions based on the initial text and emotion.
        """
        # --- THIS IS THE KEY UPDATE ---
        # We now build a more detailed emotional context for the prompt.
        
        primary_emotion = emotion_data.get('emotion', 'neutral')
        primary_score = emotion_data.get('confidence_score', 0)
        all_scores = emotion_data.get('all_scores', {})
        
        # Define a threshold for what we consider a "significant" secondary emotion.
        EMOTION_THRESHOLD = 0.10 # 10% confidence

        # Construct a detailed string describing the emotional analysis.
        emotion_details_string = f"The primary emotion detected is '{primary_emotion}' with a confidence of {primary_score * 100:.1f}%."
        
        # Find other emotions that are above our threshold.
        other_emotions = [
            f"{name} ({score * 100:.1f}%)" 
            for name, score in all_scores.items() 
            if name != primary_emotion and score > EMOTION_THRESHOLD
        ]
        
        if other_emotions:
            emotion_details_string += f" Other notable emotions include: {', '.join(other_emotions)}."

        # The search query remains focused on the primary emotion and user text for best retrieval results.
        search_query = f"Emotion: {primary_emotion}. User statement: {original_text}"
        
        retrieved_context = self._search_knowledge_base(search_query)
        context_str = "\n\n---\n\n".join(retrieved_context)

        # The prompt is now updated with the rich emotional context.
        prompt = f"""
        You are a therapeutic assistant. A user has shared this: '{original_text}'.
        Here is the emotional analysis of their statement: {emotion_details_string}
        Based on this emotional context, and the general knowledge that '{context_str}', your goal is to understand their state better.

        Generate a JSON object containing a single key "questions" which is a list of 5 to 7 clarifying questions.
        Each question must be answerable on a scale of 1 to 10.
        The questions should probe deeper into the user's statement and feelings, considering all the detected emotions.
        """
        # --- END OF THE UPDATE ---
        
        try:
            response = self.llm_model.generate_content(prompt)
            json_string = response.text.strip().replace("```json", "").replace("```", "")
            questions_data = json.loads(json_string)
            return questions_data.get("questions", [])
        except (json.JSONDecodeError, AttributeError, KeyError) as e:
            print(f"❌ Error parsing questions from LLM: {e}")
            # Fallback questions if the LLM fails
            return [
                "On a scale of 1-10, how intense is this feeling right now?",
                "On a scale of 1-10, how much does this interfere with your daily life?",
                "On a scale of 1-10, how optimistic do you feel about the future?",
                "On a scale of 1-10, how connected do you feel to others?",
                "On a scale of 1-10, how much energy do you have for things you usually enjoy?"
            ]

    def generate_summary_response(self, conversation_history: str):
        """
        Generates a final summary based on the full conversation.
        """
        prompt = f"""
        You are an empathetic therapeutic assistant. You have just had the following diagnostic conversation with a user, where they rated their feelings on a scale of 1-10.

        Conversation History:
        ---
        {conversation_history}
        ---

        Based on this entire conversation, provide a final, supportive summary in a few paragraphs (300-400 words).
        Your response should:
        1. Acknowledge their initial statement and their ratings.
        2. Validate their feelings in a warm and non-judgmental tone.
        3. Synthesize the information to offer three or four key insights into their situation.
        4. Provide two to three gentle, actionable suggestions they could try.
        """
        response = self.llm_model.generate_content(prompt)
        return response.text.strip()
