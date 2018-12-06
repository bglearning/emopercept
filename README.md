# EmoPercept

Emotion classification of text.

## Setup

Go to the project root and run

```
pipenv install
```

## Steps

Download data

```
python src/download.py
```

Train model

```
python src/train.py
```

Run Flask app

```
python app.py
```

Go to `http://localhost:5000` and know your emotions!
