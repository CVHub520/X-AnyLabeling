import fire

from anylabeling.services.auto_labeling.utils.sahi.predict import predict


def main():
    fire.Fire(predict)


if __name__ == "__main__":
    main()
