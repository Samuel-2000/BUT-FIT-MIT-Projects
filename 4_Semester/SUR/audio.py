# audio.py
import argparse
import warnings
from audio_models import (
    train_and_evaluate_gmm,
    train_and_evaluate_kmeans,
    print_predictions
)

def main():
    parser = argparse.ArgumentParser(description="Audio classification system")
    
    # Add required folder arguments
    parser.add_argument("--train_folder", type=str, required=True,
                      help="Path to training data folder")
    parser.add_argument("--test_folder", type=str, required=True,
                      help="Path to test data folder")

    # Model selection group
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--audio_gmm", action="store_true", help="Use GMM model")
    group.add_argument("--audio_kmeans", action="store_true", help="Use KMeans model")
    
    # Other arguments
    parser.add_argument("--augment", action="store_true", 
                      help="Enable data augmentation")
    parser.add_argument("--train", action="store_true",
                      help="Force training even if model exists")

    args = parser.parse_args()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        
        if args.audio_gmm:
            predictions = train_and_evaluate_gmm(
                train_folder=args.train_folder,
                test_folder=args.test_folder,
                augment=args.augment,
                force_train=args.train
            )
        elif args.audio_kmeans:
            predictions = train_and_evaluate_kmeans(
                train_folder=args.train_folder,
                test_folder=args.test_folder,
                augment=args.augment,
                force_train=args.train
            )
        
        print_predictions(predictions)

if __name__ == "__main__": 
    main()