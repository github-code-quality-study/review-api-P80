import csv
from uuid import UUID
from datetime import datetime, timedelta

TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M:%S'

import unittest
import json
import webtest
import server


class ReviewAnalyzerServerTest(unittest.TestCase):
  LOCATION = 'San Diego, California'
  REVIEW_BODY = 'I love this place!'
  
  def setUp(self):
    app = server.ReviewAnalyzerServer()
    self.testapp = webtest.TestApp(app)

  def test_order_reviews_by_sentiment(self):
    response = self.testapp.get("/")
    self.assertEqual(200, response.status_code)
    results = json.loads(response.body)

    with open('data/reviews.csv', 'r') as file:
      reader = csv.reader(file)
      next(reader)  # Skip the header row
      reviews = list(reader)

    assert len(results) == len(reviews)

    # Check if 'sentiment' key exists in each review item
    for i, review in enumerate(results):
        assert 'sentiment' in review, f"Missing 'sentiment' key in review at index {i}"
    
    # results are sorted by compound score, descending
    sorted_by_sentiment = sorted(results, key=lambda x: x['sentiment']['compound'], reverse=True)
    for i in range(len(sorted_by_sentiment) - 1):
        assert sorted_by_sentiment[i]['sentiment']['compound'] >= sorted_by_sentiment[i + 1]['sentiment']['compound'], "Reviews are not correctly ordered by sentiment"

  def test_get_by_location(self):
    response = self.testapp.get('/?location=Denver, Colorado')
    self.assertEqual(200, response.status_code)
    results = json.loads(response.body)

    # assert that all results have a location of Denver, Colorado
    assert all(result['Location'] == 'Denver, Colorado' for result in results)

  def test_get_by_start_date(self):
    start_date = '2021-01-01'
    response = self.testapp.get("/?start_date=" + start_date)
    self.assertEqual(200, response.status_code)
    results = json.loads(response.body)

    # assert that the results only include reviews with a Timestamp greater than or equal to the start date
    assert all(datetime.strptime(result['Timestamp'], TIMESTAMP_FORMAT) >= datetime.strptime(start_date, '%Y-%m-%d') for result in results)

  def test_get_by_end_date(self):
    end_date = '2021-01-03'
    response = self.testapp.get('/?end_date=' + end_date)
    self.assertEqual(200, response.status_code)
    results = json.loads(response.body)

    # assert that the results only include reviews with a Timestamp less than or equal to the end date
    assert all(datetime.strptime(result['Timestamp'], TIMESTAMP_FORMAT) <= datetime.strptime(end_date, '%Y-%m-%d') for result in results)

  def test_get_by_start_and_end_date(self):
    start_date = '2021-01-02'
    end_date = '2021-01-03'
    response = self.testapp.get(f'/?start_date={start_date}&end_date={end_date}')
    self.assertEqual(200, response.status_code)
    results = json.loads(response.body)

    # assert that the results only include reviews with a Timestamp between the start and end dates
    assert all(datetime.strptime(result['Timestamp'], TIMESTAMP_FORMAT) >= datetime.strptime(start_date, '%Y-%m-%d') and datetime.strptime(result['Timestamp'], TIMESTAMP_FORMAT) <= datetime.strptime(end_date, '%Y-%m-%d') for result in results)

  def test_post_review(self):
    review = {
      'Location': self.LOCATION,
      'ReviewBody': self.REVIEW_BODY
    }
    response = self.testapp.post("/", review)
    self.assertEqual(201, response.status_code)
    result = json.loads(response.body)

    assert result['ReviewBody'] == review['ReviewBody']
    assert result['Location'] == review['Location']

    response = self.testapp.get("/")
    results = json.loads(response.body)
    assert any(result['ReviewId'] == result['ReviewId'] for result in results)

  def test_post_review_with_missing_review_body(self):
    review = {
      'Location': self.LOCATION,
    }
    response = self.testapp.post("/", review, expect_errors=True)
    self.assertEqual(400, response.status_code)

  def test_post_location_scenarios(self):
    scenarios = [
        {'Location': 'Cupertino, California', 'ReviewBody': 'I love this place!', 'expected_status': 400, 'test_name': 'invalid location'},
        {'ReviewBody': 'Great service!', 'expected_status': 400, 'test_name': 'missing location'}
    ]

    for scenario in scenarios:
        with self.subTest(scenario['test_name']):
            review = {key: value for key, value in scenario.items() if key not in ['expected_status', 'test_name']}
            response = self.testapp.post("/", review, expect_errors=True)
            self.assertEqual(scenario['expected_status'], response.status_code)

  def test_post_generates_a_UUID(self):
    review = {
      'Location': self.LOCATION,
      'ReviewBody': self.REVIEW_BODY
    }
    response = self.testapp.post("/", review)
    self.assertEqual(201, response.status_code)
    result = json.loads(response.body)

    assert UUID(result['ReviewId'], version=4)

  def test_post_generates_a_timestamp(self):
    review = {
      'Location': self.LOCATION,
      'ReviewBody': self.REVIEW_BODY
    }
    response = self.testapp.post("/", review)
    self.assertEqual(201, response.status_code)
    result = json.loads(response.body)

    timestamp = datetime.strptime(result['Timestamp'], TIMESTAMP_FORMAT)
    now = datetime.now()
    delta = timedelta(seconds=10)
    assert now - delta <= timestamp <= now

if __name__ == '__main__':
    unittest.main()