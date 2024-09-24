[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=15545315)
# Instructions
You are being asked to complete `GET` and `POST` endpoints for the restaurant reviews API of a fictional restaurant chain. For this study you’ll be asked to load the reviews from `data/reviews.csv` into memory to complete the tasks. You are asked to complete the `GET` and `POST` methods for the server according to the specifications outlined below.

The reviews will need to be analyzed for sentiment using the method `analyze_sentiment`; this method will not need to be updated or edited in any way and can be assumed to work as intended. The `analyze_sentiment` method returns a dictionary with four items: `'neg'`, `'neu'`, `'pos'`, and `'compound'`. These represent the negative, neutral, positive, and compound sentiment scores of the input text, respectively.

## Getting started
To get started, install the additional libraries by running:

```Bash
pip install -r requirements.txt
```

To start the server run:

```Bash
python server.py
```

> [!IMPORTANT]
> Your applicaiton will be available at URL `https://CODESPACENAME-PORT.app.github.dev` Use that URL for GET requests. For more information see [Forwarding Ports in your Codespace](https://docs.github.com/en/codespaces/developing-in-a-codespace/forwarding-ports-in-your-codespace#using-command-line-tools-and-rest-clients-to-access-ports-1)

When you start the server, you should see a pop up window in the lower right conrner saying that your application is running on port 8000. If you open that link (`https://CODESPACENAME-8000.app.github.dev`) you will see a response similar to:


```JSON
[
  {
    "ReviewId": "95d31055-4d0a-4e7f-9e31-2d118aabf8c0",
    "Location": "Denver, Colorado",
    "Timestamp": "2016-02-16 14:16:33",
    "ReviewBody": "As we ate, we noticed that the restaurant was also very family-friendly. There were high chairs and booster seats available, and the staff was happy to bring out crayons and coloring pages for the kids."
  },
  {
    "ReviewId": "20e0c5cc-2e1e-4cdc-8d86-cf43b8c44fe8",
    "Location": "Salt Lake City, Utah",
    "Timestamp": "2016-05-20 6:08:52",
    "ReviewBody": "Craving chicken? Jimmy's Chicken has got you covered. Affordable prices, tasty dishes, generous portions, friendly staff, cozy atmosphere."
  },
  ... and so on for all the reviews
]
```

> [!NOTE]  
> The exact reviews may differ, but yous should see a JSON response with a list of reviews. If you do not your server is not running correctly or you are not connected to the correct URL.

## GET 
The GET method should allow for two parameters `Location` and `Timestamp` that filters the results when the parameters are included. GET should return a JSON object with the following structure:
```JSON
{
    "ReviewId": "<UUID of the review>",
    "ReviewBody": "<Text of the review>",
    "Location": "<Location associated with the review>",
    "Timestamp": "<Time the review was submitted>",
    "sentiment": {
        "neg": "<Negative sentiment score>",
        "neu": "<Neutral sentiment score>",
        "pos": "<Positive sentiment score>",
        "compound": "<Compound sentiment score>"
    }
}
```
The results should be returned in descending order by the `compound` value in sentiment. 

### Location Parameter

The server should only return results filtered by location for these locations:

- Albuquerque, New Mexico
- Carlsbad, California
- Chula Vista, California
- Colorado Springs, Colorado
- Denver, Colorado
- El Cajon, California
- El Paso, Texas
- Escondido, California
- Fresno, California
- La Mesa, California
- Las Vegas, Nevada
- Los Angeles, California
- Oceanside, California
- Phoenix, Arizona
- Sacramento, California
- Salt Lake City, Utah
- Salt Lake City, Utah
- San Diego, California
- Tucson, Arizona

### Timestamp Parameter

To view the reviews for a specific date range you can include the `start_date` and `end_date` parameters in the URL.

### Example GET requests
The server should accommodate GET requests of the following format:

```URL
https://CODESPACENAME-port.app.github.dev/?location=LOCATION&start_date=YYYY-MM-DD&end_date=YYYY-MM-DD`
```
An example request is: 

```URL
https://CODESPACENAME-8000.app.github.dev/?location=Salt+Lake+City%2C+Utah&start_date=2021-01-01&end_date=2021-12-31
```
Returns:
```JSON
[
  {
    "ReviewId": "dd578dea-2616-4b4c-a280-83fee7ee2688",
    "Location": "Salt Lake City, Utah",
    "Timestamp": "2021-01-23 2:35:55",
    "ReviewBody": "Jimmy's Chicken is the go-to spot for a family meal that won't disappoint. Friendly staff, delicious chicken dishes, and affordable prices make it a great value.",
    "sentiment": {
      "neg": 0.077,
      "neu": 0.557,
      "pos": 0.365,
      "compound": 0.8699
    }
  },
  {
    "ReviewId": "63f2f45a-df71-4fef-b9c9-7b11e4dd540a",
    "Location": "Salt Lake City, Utah",
    "Timestamp": "2021-06-26 13:01:38",
    "ReviewBody": "Jimmy's Chicken is the go-to spot for a family meal that won't disappoint. Friendly staff, delicious chicken dishes, and affordable prices make it a great value.",
    "sentiment": {
      "neg": 0.077,
      "neu": 0.557,
      "pos": 0.365,
      "compound": 0.8699
    }
  },
  {
    "ReviewId": "11088493-d95d-43b5-beed-5552725a0835",
    "Location": "Salt Lake City, Utah",
    "Timestamp": "2021-10-21 6:09:40",
    "ReviewBody": "The quality of food at Jimmy's Chicken is top-notch!",
    "sentiment": {
      "neg": 0,
      "neu": 1,
      "pos": 0,
      "compound": 0
    }
  }
]
```

## POST 
Post should accept two parameters, `ReviewBody` and `Location` , both are text strings. Each post should add a Timestamp for when the review was added using `datetime` and a ReviewId using `uuid` which are already included in the import statements.

> [!IMPORTANT] 
> Unlike the GET requests above, you do not need to direct POST requests to the Codespace. Instead use `http://localhost:8000`


### Example Post

```Bash
curl -X POST http://localhost:8000 -d "Location=San Diego, California&ReviewBody=I love this place!"
```

Should return: 
```JSON
{"ReviewId": "72e401ba-edd2-41c9-87ae-589637b781c7", "ReviewBody": "I love this place!", "Location": "San Diego, California", "Timestamp": "2024-05-13 11:21:30"}
``` 

## Tests
Write your server implementation in `server.py`. Then run the tests:
```Bash
python -m unittest server_test.py
```
You can run individual tests by running the command: 
```Bash
python server_test.py ReviewAnalyzerServerTest.<test_name>
```

## What to submit 

When you are finished with you work you can commit your changes and push you work with the following command:
```Bash
git commit -am "Your commit message" && git push origin main
```

You can use the UI to commit and push your changes as well, but please ensure that all your changes are pushed to the repository. 
