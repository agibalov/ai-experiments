from openai import OpenAI
from pydantic import BaseModel
from enum import Enum

def test_sentiment_analysis():
    class Sentiment(str, Enum):
        positive = "positive"
        neutral = "neutral"
        negative = "negative"

    class SentimentAnalysisResponse(BaseModel):
        sentiment: Sentiment

    client = OpenAI()
    def sentiment(text: str) -> SentimentAnalysisResponse:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "system",
                    "content": """
                    Perform sentiment analysis of a product review.

                    Sentiment is "positive" only if text states good things and doesn't
                    state bad things. If there's even a tiniest bad thing mentioned
                    in the text, the sentiment is not "positive".

                    Sentiment is "negative" only if text states bad things and doesn't
                    state good things. If there's even a tiniest good thing mentioned
                    in the text, the sentiment is not "negative".

                    Sentiment is "neutral" if it's not "positive" and not "negative".
                    """
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            response_format=SentimentAnalysisResponse
        )
        return completion.choices[0].message.parsed.sentiment

    assert sentiment("""Build quality is legendary! Love the traditional color
                   of course, easy to use , clean. Light weight but feels
                   substantial. A great quality. Zero leaks and the temperature
                   was astounding.""") == Sentiment.positive

    assert sentiment("""Just terrible quality product. Bought this a couple years
                   back, the cap disintegrated after a few months of use. Called
                   and after several weeks and submitting bunch of pictures and
                   complaints, they sent another cap, still plastic. ) months
                   later we are on the same page again. They save maybe 10 cents
                   making this out of metal. What a waste of
                   a good brand.""") == Sentiment.negative

    assert sentiment("""I want to believe they've learned to make as good or even
                     a better thermos out of lighter/better steel, but you can feel
                     the weight difference between this and my 30 year old Stanley
                     bottle immediately. They still keep coffee/water hot for days,
                     so long as you perform the hot water pre-rinse. I just hope
                     they'll hold up to the beating I put into my
                     last one.""") == Sentiment.neutral
