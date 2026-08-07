import matplotlib.pyplot as plt
from wordcloud import WordCloud


class Visualizer:

    def __init__(self, dataset):
        self.dataset = dataset

    # Category Distribution
    def plot_category_distribution(self):

        plt.figure(figsize=(10,6))

        self.dataset["Category"].value_counts().plot(kind="bar")

        plt.title("Resume Category Distribution")
        plt.xlabel("Category")
        plt.ylabel("Count")
        plt.xticks(rotation=45)

        plt.tight_layout()

        plt.savefig("Visualizations/category_distribution.png")

        plt.show()

        print("Category Distribution Saved.")

    # Word Cloud
    def word_cloud(self):

        text = " ".join(self.dataset["Resume"])

        cloud = WordCloud(
            width=1000,
            height=500,
            background_color="white"
        ).generate(text)

        plt.figure(figsize=(12,6))
        plt.imshow(cloud)
        plt.axis("off")

        plt.savefig("Visualizations/wordcloud.png")

        plt.show()

        print("Word Cloud Saved.")