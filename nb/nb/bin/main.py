import argparse

from nb.nb import NaiveBayes
from nb.utils.load_data import build_dataframe
from nb.utils.metrics import get_metrics


def main():
    parser = argparse.ArgumentParser(description="Naive Bayes Algorithm")
    parser.add_argument("-f", "--indir", required=True, help="Data directory")
    parser.add_argument("-a", "--alpha", type=float, default=0.1,
                        help="Lidstone smoothing parameter")
    args = parser.parse_args()

    # Build the DataFrames from the text within the data subfolders
    training_df, test_df = build_dataframe(args.indir)

    # Estimate priors and likelihoods by training the model
    model = NaiveBayes(alpha=args.alpha).train(training_df)

    # Run the model against the unlabeled speeches
    predictions = model.test(test_df)

    print("Predictions:")
    for label, pred in zip(test_df["author"], predictions):
        print(f"  true: {label}  predicted: {pred}")

    acc, f1, conf = get_metrics(test_df["author"], predictions)
    print("\nNaive Bayes")
    print(f"  accuracy: {acc:.3f}")
    print(f"  f1 score: {f1:.3f}")
    print(f"  confusion matrix:\n{conf}")


if __name__ == "__main__":
    main()
