from predict import Predictor

predictor = Predictor()
predictor.setup()

result = predictor.predict(
    video="/Users/b/Downloads/prn.mp4",
    prompt="What is happening in this video?"
)
print("Result:", result)