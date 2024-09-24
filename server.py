import csv
import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Any, Callable, List, Dict
from urllib.parse import parse_qs, urlparse
from wsgiref.simple_server import make_server

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Configure NLTK
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

# Constants
TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M:%S'
VALID_LOCATIONS = [
    'Albuquerque, New Mexico', 'Carlsbad, California', 'Chula Vista, California',
    'Colorado Springs, Colorado', 'Denver, Colorado', 'El Cajon, California',
    'El Paso, Texas', 'Escondido, California', 'Fresno, California',
    'La Mesa, California', 'Las Vegas, Nevada', 'Los Angeles, California',
    'Oceanside, California', 'Phoenix, Arizona', 'Sacramento, California',
    'Salt Lake City, Utah', 'San Diego, California', 'Tucson, Arizona'
]

# Initialize NLTK's sentiment analyzer
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

# Read reviews from CSV file
reviews = pd.read_csv('data/reviews.csv').to_dict('records')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        """Initialize the ReviewAnalyzerServer."""
        pass

    def analyze_sentiment(self, review_body: str) -> Dict[str, float]:
        """Analyze the sentiment of the given review."""
        return sia.polarity_scores(review_body)

    def filter_reviews(self, query_params: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Filter reviews based on query parameters."""
        location = query_params.get("location", [""])[0]
        start_date = query_params.get("start_date", [""])[0]
        end_date = query_params.get("end_date", [""])[0]

        filtered_reviews = []
        for review in reviews:
            if location and location != review["Location"]:
                continue

            if start_date or end_date:
                review_date = datetime.strptime(review["Timestamp"], TIMESTAMP_FORMAT)
                if start_date and review_date < datetime.strptime(start_date, "%Y-%m-%d"):
                    continue
                if end_date and review_date > datetime.strptime(end_date, "%Y-%m-%d"):
                    continue

            review_with_sentiment = review.copy()
            review_with_sentiment["sentiment"] = self.analyze_sentiment(review["ReviewBody"])
            filtered_reviews.append(review_with_sentiment)

        return sorted(filtered_reviews, key=lambda x: x["sentiment"]["compound"], reverse=True)

    def handle_get(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """Handle GET requests."""
        query_params = parse_qs(environ["QUERY_STRING"])
        filtered_reviews = self.filter_reviews(query_params)

        response_body = json.dumps(filtered_reviews, indent=2).encode("utf-8")
        start_response("200 OK", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(response_body)))
        ])
        return [response_body]

    def handle_post(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """Handle POST requests."""
        try:
            request_body_size = int(environ.get("CONTENT_LENGTH", 0))
            request_body = environ["wsgi.input"].read(request_body_size).decode("utf-8")
            post_data = parse_qs(request_body)

            location = post_data.get("Location", [""])[0]
            review_body = post_data.get("ReviewBody", [""])[0]

            if not location or not review_body:
                raise ValueError("Location and ReviewBody are required.")
            
            if location not in VALID_LOCATIONS:
                raise ValueError("Invalid location provided.")

            new_review = self.create_review(location, review_body)

            response_body = json.dumps(new_review).encode("utf-8")
            start_response("201 Created", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            return [response_body]

        except Exception as e:
            logger.error(f"Error handling POST request: {e}")
            start_response("400 Bad Request", [
                ("Content-Type", "application/json")
            ])
            return [json.dumps({"error": str(e)}).encode("utf-8")]

    def create_review(self, location: str, review_body: str) -> Dict[str, Any]:
        """Create a new review entry."""
        review_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)

        new_review = {
            "ReviewId": review_id,
            "Location": location,
            "Timestamp": timestamp,
            "ReviewBody": review_body,
            "sentiment": self.analyze_sentiment(review_body)
        }

        reviews.append(new_review)
        return new_review

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """Handle incoming requests."""
        if environ["REQUEST_METHOD"] == "GET":
            return self.handle_get(environ, start_response)

        if environ["REQUEST_METHOD"] == "POST":
            return self.handle_post(environ, start_response)

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = int(os.environ.get('PORT', 8000))
    with make_server("", port, app) as httpd:
        logger.info(f"Listening on port {port}...")
        httpd.serve_forever()
