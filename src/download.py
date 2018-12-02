import wget


def main():
    wget.download('https://drive.google.com/uc?export=download&id=1nVp1XD9IVA3CGKcUmegC8Pm1dd-pUmfE', 
            "data/emotions.csv")


if __name__ == '__main__':
    main()
