import torch
import torch.nn as nn
import torch.nn.functional as F

class LargeMarginLoss(nn.Module):
    def __init__(self, num_classes, margin=0.2, scale=30.0):
        super(LargeMarginLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

    def forward(self, embeddings, labels):
        # Compute pairwise cosine similarity between embeddings
        similarity_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
        
        # Gather the similarity values for positive and negative pairs
        pos_pairs = similarity_matrix[labels.unsqueeze(1) == labels.unsqueeze(0)]
        neg_pairs = similarity_matrix[labels.unsqueeze(1) != labels.unsqueeze(0)]
        
        # Compute loss using large margin loss formula
        loss = torch.mean(torch.log(1 + torch.sum(torch.exp(self.scale * (neg_pairs - self.margin))) / torch.sum(torch.exp(self.scale * (pos_pairs - self.margin)))))
        
        return loss


# Example usage:
# Create an instance of the LargeMarginLoss
num_classes = 100  # Replace with the number of classes in your task
margin_loss = LargeMarginLoss(num_classes)

# Compute the loss given the model's embeddings and labels
embeddings = torch.randn((batch_size, embedding_dim), requires_grad=True)  # Replace with your actual embeddings
labels = torch.randint(0, num_classes, (batch_size,))
loss = margin_loss(embeddings, labels)
