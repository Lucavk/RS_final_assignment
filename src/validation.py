from src.models.als import ALSRecommender
from src.models.itemknn import ItemKNNRecommender
from src.models.popularity import PopularityRecommender
from src.metrics import RankingMetrics
from src.data import (
    InteractionDataLoader,
    TemporalLeaveOneOutSplitter,
    UserHistoryBuilder
)
from src.config import Config
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


class ValidationRunner:
    def __init__(self, config):
        self.config = config

    def run(self):
        print("Loading data...")
        loader = InteractionDataLoader(self.config.TRAIN_PATH)
        df = loader.load_train()

        print()
        print("Dataset info:")
        print(f"Rows:  {len(df):,}")
        print(f"Users: {df['user_id'].nunique():,}")
        print(f"Items: {df['item_id'].nunique():,}")

        print()
        print("Creating temporal leave-one-out validation split...")
        splitter = TemporalLeaveOneOutSplitter()
        train_split, val_targets = splitter.split(df)

        print(f"Training rows after split: {len(train_split):,}")
        print(f"Validation users:          {len(val_targets):,}")

        print()
        print("Building user histories and seen-item dictionary...")
        user_histories = UserHistoryBuilder.build_user_histories(train_split)
        user_seen_items = UserHistoryBuilder.build_user_seen_items(train_split)

        results = {}

        print()
        print("=" * 60)
        print("Training popularity baseline...")
        print("=" * 60)

        popularity_model = PopularityRecommender()
        popularity_model.fit(train_split)

        popularity_score = RankingMetrics.recall_at_k(
            model=popularity_model,
            val_targets=val_targets,
            user_histories=user_histories,
            user_seen_items=user_seen_items,
            k=self.config.K
        )

        results["Popularity"] = popularity_score

        print()
        print(f"Popularity Recall@{self.config.K}: {popularity_score:.6f}")

        print()
        print("=" * 60)
        print("Training ItemKNN co-occurrence model...")
        print("=" * 60)

        itemknn_model = ItemKNNRecommender(
            max_history_items=self.config.MAX_HISTORY_ITEMS,
            max_neighbors_per_item=self.config.MAX_NEIGHBORS_PER_ITEM,
            min_cooc_count=self.config.MIN_COOC_COUNT,
            fallback_model=popularity_model
        )

        itemknn_model.fit(train_split)

        itemknn_score = RankingMetrics.recall_at_k(
            model=itemknn_model,
            val_targets=val_targets,
            user_histories=user_histories,
            user_seen_items=user_seen_items,
            k=self.config.K
        )

        results["ItemKNN"] = itemknn_score

        print()
        print(f"ItemKNN Recall@{self.config.K}: {itemknn_score:.6f}")

        print()
        print("=" * 60)
        print("Training ALS matrix factorization model...")
        print("=" * 60)

        als_model = ALSRecommender(
            factors=self.config.ALS_FACTORS,
            regularization=self.config.ALS_REGULARIZATION,
            iterations=self.config.ALS_ITERATIONS,
            alpha=self.config.ALS_ALPHA,
            fallback_model=popularity_model,
            random_state=42
        )

        als_model.fit(train_split)

        als_score = RankingMetrics.recall_at_k(
            model=als_model,
            val_targets=val_targets,
            user_histories=user_histories,
            user_seen_items=user_seen_items,
            k=self.config.K
        )

        results["ALS"] = als_score

        print()
        print(f"ALS Recall@{self.config.K}: {als_score:.6f}")

        print()
        print("=" * 60)
        print("Validation results")
        print("=" * 60)

        for model_name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
            print(f"{model_name:<12} Recall@{self.config.K}: {score:.6f}")

        best_model_name, best_score = max(results.items(), key=lambda x: x[1])

        print()
        print(f"Best model so far: {best_model_name}")
        print(f"Best Recall@{self.config.K}: {best_score:.6f}")

        print()
        print("Comparison against popularity:")

        popularity_score = results["Popularity"]

        for model_name, score in results.items():
            if model_name == "Popularity":
                continue

            absolute_improvement = score - popularity_score
            relative_improvement = (
                absolute_improvement / popularity_score * 100
                if popularity_score > 0
                else 0
            )

            print(
                f"{model_name:<12} absolute improvement: {absolute_improvement:.6f} | "
                f"relative improvement: {relative_improvement:.2f}%"
            )


def main():
    runner = ValidationRunner(Config)
    runner.run()


if __name__ == "__main__":
    main()
