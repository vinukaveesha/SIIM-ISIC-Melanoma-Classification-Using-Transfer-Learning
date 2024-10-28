import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder, normalize


class MelanomaDataset:
    def __init__(self, directory):
        self.directory = directory
        self.train_df = pd.read_csv(os.path.join(directory, "train.csv"))
        self.colors_nude = [
            "#FFC1CC",
            "#FFD1DC",
            "#FF69B4",
            "#FF6EB4",
            "#FF1493",
        ]

    def count_files(self, sub_dir, extension):
        path = os.path.join(self.directory, sub_dir)
        count = sum(1 for file in os.listdir(path) if file.endswith(extension))
        print(f"{extension} files in {sub_dir}: {count}")
        return count

    def rename_cloumns(self):
        self.train_df.rename(
            columns={
                "patient_id": "Id",
                "age_approx": "Age",
                "anatom_site_general_challenge": "Anatomy",
                "benign_malignant": "is_malignant",
            },
            inplace=True,
        )

    def fill_missing_sex(self):
        self.train_df["sex"] = self.train_df["sex"].fillna("male")
        return self.train_df

    def fill_missing_age(self):
        median_age = self.train_df["Age"].median()
        self.train_df["Age"] = self.train_df["Age"].fillna(median_age)
        return self.train_df

    def fill_missing_anatomy(self):
        self.train_df["Anatomy"] = self.train_df["Anatomy"].fillna("torso")
        return self.train_df

    def encode_features(self):
        encoder = LabelEncoder()
        features_to_encode = ["sex", "Anatomy", "diagnosis"]
        for feature in features_to_encode:
            self.train_df[feature] = encoder.fit_transform(self.train_df[feature])
        return self.train_df

    def normalize_features(self):
        features = ["sex", "Age", "Anatomy"]
        normalized_data = normalize(self.train_df[features])
        self.train_df[features] = normalized_data
        return self.train_df

    def save_dataframe(self, filename="new_train.csv"):
        self.train_df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")

    def process_and_save(self):
        self.rename_cloumns()
        self.fill_missing_sex()
        self.fill_missing_age()
        self.fill_missing_anatomy()
        self.save_plots()
        self.encode_features()
        self.normalize_features()
        self.save_dataframe()

    def save_plots(self):

        # Setting up a directory for plots
        plots_directory = "results/plots"
        os.makedirs(plots_directory, exist_ok=True)
        df = self.train_df

        # Gender split by target variable
        plt.figure(figsize=(12, 6))
        sns.countplot(
            x="target",
            hue="sex",
            data=df,
            palette=self.colors_nude[2:4],
        )
        plt.title("Gender Split by Target Variable")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{plots_directory}/gender_split_by_target.png")
        plt.close()

        # Age distribution by target status
        plt.figure(figsize=(12, 6))
        sns.kdeplot(
            data=df[df["target"] == 0]["Age"],
            color=self.colors_nude[2],
            label="Benign",
            linewidth=2,
        )
        sns.kdeplot(
            data=df[df["target"] == 1]["Age"],
            color=self.colors_nude[3],
            label="Malignant",
            linewidth=2,
        )
        plt.title("Age Distribution by Target Status")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{plots_directory}/age_distribution_by_target.png")
        plt.close()

        # Anatomy frequencies (showing top 10 most frequent anatomies)
        top_anatomies = df["Anatomy"].value_counts().nlargest(10).index
        plt.figure(figsize=(12, 6))
        sns.countplot(
            x="Anatomy",
            data=df[df["Anatomy"].isin(top_anatomies)],
            palette=self.colors_nude,
        )
        plt.title("Top 10 Anatomy Frequencies")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{plots_directory}/anatomy_frequencies.png")
        plt.close()

        # Diagnosis frequencies
        plt.figure(figsize=(12, 6))
        sns.countplot(x="diagnosis", data=df, palette=self.colors_nude)
        plt.title("Diagnosis Frequencies")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{plots_directory}/diagnosis_frequencies.png")
        plt.close()

        # Diagnosis split by target variable
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        sns.countplot(
            x="diagnosis",
            data=df[df["target"] == 0],
            palette=self.colors_nude,
            ax=ax1,
        )
        ax1.set_title("Diagnosis Frequencies for Benign Cases")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)

        sns.countplot(
            x="diagnosis",
            data=df[df["target"] == 1],
            palette=self.colors_nude,
            ax=ax2,
        )
        ax2.set_title("Diagnosis Frequencies for Malignant Cases")
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)

        plt.tight_layout()
        plt.savefig(f"{plots_directory}/diagnosis_split_by_target.png")
        plt.close()
