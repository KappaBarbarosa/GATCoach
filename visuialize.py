import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def load_and_visualize_embeddings(directory, pattern='*.pt'):
    """Load and visualize embeddings from saved checkpoint files."""
    print(directory)
    print (pattern)
    file_paths = glob.glob(os.path.join(directory, pattern))
    
    print (file_paths)

    for file_path in file_paths:
        embeddings = torch.load(file_path).detach().cpu().numpy()
        print(embeddings.shape)
        for i in range(embeddings.shape[0]):
            print(embeddings[i][:6])
        # Use perplexity 3 and 1000 iterations for t-SNE
        embeddings_tsne = TSNE(n_components=2, perplexity=3, n_iter=1000).fit_transform(embeddings)
        
        # Create the plot
        fig, ax = plt.subplots()
        ax.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1])
        
        # Annotate each point with its corresponding strategy number
        for i, point in enumerate(embeddings_tsne):
            ax.annotate(str(i + 1), (point[0], point[1]), textcoords="offset points", xytext=(0,10), ha='center')
        
        ax.set_title(os.path.basename(file_path))
        # Save the plot by replacing the '.pt' file extension with '.png'
        output_filename = os.path.splitext(file_path)[0] + '.png'
        plt.savefig(output_filename)
        plt.close()
        print(f"Visualization saved as {output_filename}")

if __name__ == '__main__':
    load_and_visualize_embeddings('visuialize/embeddings/gnn n_strategy: 4, uniform, 2s3z')
