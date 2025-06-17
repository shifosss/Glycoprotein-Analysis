import re
import os
import pickle as pkl
from typing import List, Optional, Dict, Sequence
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_scatter import scatter_mean

try:
    from glycowork.motif import tokenization

    GLYCOWORK_AVAILABLE = True
except ImportError:
    GLYCOWORK_AVAILABLE = False
    print("Warning: glycowork not available. Some features may be limited.")

# Import readout classes
from embedder.GlycanEmbedder_Package.readout import MeanReadout, SumReadout, MaxReadout, AttentionReadout


# ============================================================================
# Utility Functions
# ============================================================================

activation_map = {
                        'relu': 'ReLU',
                        'gelu': 'GELU', 
                        'tanh': 'Tanh',
                        'sigmoid': 'Sigmoid',
                        'leakyrelu': 'LeakyReLU'
                    }

def variadic_to_padded(input_tensor, sizes, padding_value=0):
    """Convert variadic tensor to padded tensor."""
    max_size = sizes.max().item()
    batch_size = len(sizes)
    
    # Handle both 1D and 2D+ tensors
    if input_tensor.dim() == 1:
        # For 1D tensors (token IDs), create 2D output
        output = torch.full((batch_size, max_size),
                            padding_value, dtype=input_tensor.dtype, device=input_tensor.device)
    else:
        # For 2D+ tensors (embeddings), preserve the last dimension
        output = torch.full((batch_size, max_size, input_tensor.size(-1)),
                            padding_value, dtype=input_tensor.dtype, device=input_tensor.device)

    offset = 0
    for i, size in enumerate(sizes):
        output[i, :size] = input_tensor[offset:offset + size]
        offset += size

    mask = torch.arange(max_size, device=input_tensor.device).expand(batch_size, -1) < sizes.unsqueeze(1)
    return output, mask


def padded_to_variadic(padded_tensor, sizes):
    """Convert padded tensor back to variadic tensor."""
    output_list = []
    for i, size in enumerate(sizes):
        output_list.append(padded_tensor[i, :size])
    return torch.cat(output_list, dim=0)


# ============================================================================
# Base Embedder Class
# ============================================================================

class BaseGlycanEmbedder(nn.Module):
    """Base class for all glycan embedders."""

    def __init__(self):
        super().__init__()
        self.output_dim = None
        self.node_output_dim = None

    def get_output_dim(self):
        """Get the output dimension of graph-level features."""
        return self.output_dim

    def get_node_output_dim(self):
        """Get the output dimension of node-level features."""
        return self.node_output_dim or self.output_dim


# ============================================================================
# Graph-based Embedders
# ============================================================================

class GlycanGCN(BaseGlycanEmbedder):
    """
    Graph Convolutional Network for glycan embedding.

    Parameters:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        num_unit (int): number of monosaccharide units
        edge_input_dim (int, optional): dimension of edge features
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        readout (str, optional): readout function. Available functions are ``sum``, ``mean``, ``max``, ``attention``, and ``dual``.
    """

    def __init__(self, input_dim, hidden_dims, num_unit, edge_input_dim=None, short_cut=False,
                 batch_norm=False, activation="relu", concat_hidden=False, readout="mean"):
        super().__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]

        self.input_dim = input_dim
        self.dims = [input_dim] + list(hidden_dims)
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.num_unit = num_unit

        # Embedding layer
        self.embedding = nn.Embedding(num_unit, input_dim)

        # Graph convolution layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None

        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(self.dims[i + 1]))

        # Activation
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        # Readout
        if readout == "sum":
            self.readout = SumReadout()
        elif readout == "mean":
            self.readout = MeanReadout()
        elif readout == "max":
            self.readout = MaxReadout()
        elif readout == "attention":
            readout_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1)
            self.readout = AttentionReadout(readout_dim)
        elif readout == "dual":
            self.readout = MeanReadout()
            self.readout_ext = MaxReadout()

        # Output dimensions
        self.output_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1)
        if readout == "dual":
            self.output_dim = self.output_dim * 2
        self.node_output_dim = self.output_dim if readout != "dual" else self.output_dim // 2

    def forward(self, graph, input=None, all_loss=None, metric=None):
        """
        Forward pass.

        Parameters:
            graph: graph object with unit_type attribute
            input: input features (if None, uses embeddings)

        Returns:
            dict: containing "graph_feature" and "node_feature"
        """
        # Get node features from embeddings
        if input is None:
            input = self.embedding(graph.unit_type)

        hiddens = []
        layer_input = input

        # Apply graph convolution layers
        for i, layer in enumerate(self.layers):
            hidden = layer(layer_input)

            if self.batch_norms is not None:
                hidden = self.batch_norms[i](hidden)

            if i < len(self.layers) - 1:  # No activation after last layer
                hidden = self.activation(hidden)

            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input

            hiddens.append(hidden)
            layer_input = hidden

        # Concatenate hidden states if specified
        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]

        # Graph-level readout
        if hasattr(self, "readout_ext"):
            graph_feature = torch.cat([
                self.readout(graph, node_feature),
                self.readout_ext(graph, node_feature)
            ], dim=-1)
        else:
            graph_feature = self.readout(graph, node_feature)

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature
        }


class GlycanRGCN(BaseGlycanEmbedder):
    """
    Relational Graph Convolutional Network for glycan embedding.

    Parameters:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        num_unit (int): number of monosaccharide units
        num_relation (int): number of relations
        edge_input_dim (int, optional): dimension of edge features
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        readout (str, optional): readout function
    """

    def __init__(self, input_dim=128, hidden_dims=[128, 128, 128], num_unit=143, num_relation=84, edge_input_dim=None,
                 short_cut=False, batch_norm=False, activation="relu", concat_hidden=False, readout="mean"):
        super().__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]

        self.input_dim = input_dim
        self.dims = [input_dim] + list(hidden_dims)
        self.num_relation = num_relation
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

        # Embeddings
        self.embedding = nn.Embedding(num_unit, input_dim)
        self.relation_embedding = nn.Embedding(num_relation, input_dim)

        # RGCN layers (simplified)
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None

        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(self.dims[i + 1]))

        # Activation
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        # Readout
        if readout == "sum":
            self.readout = SumReadout()
        elif readout == "mean":
            self.readout = MeanReadout()
        elif readout == "max":
            self.readout = MaxReadout()
        elif readout == "dual":
            self.readout = MeanReadout()
            self.readout_ext = MaxReadout()

        # Output dimensions
        self.output_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1)
        if readout == "dual":
            self.output_dim = self.output_dim * 2
        self.node_output_dim = self.output_dim if readout != "dual" else self.output_dim // 2

    def forward(self, graph, input=None, all_loss=None, metric=None):
        if input is None:
            input = self.embedding(graph.unit_type)

        hiddens = []
        layer_input = input

        for i, layer in enumerate(self.layers):
            hidden = layer(layer_input)

            if self.batch_norms is not None:
                hidden = self.batch_norms[i](hidden)

            if i < len(self.layers) - 1:
                hidden = self.activation(hidden)

            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input

            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]

        if hasattr(self, "readout_ext"):
            graph_feature = torch.cat([
                self.readout(graph, node_feature),
                self.readout_ext(graph, node_feature)
            ], dim=-1)
        else:
            graph_feature = self.readout(graph, node_feature)

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature
        }


class GlycanGAT(BaseGlycanEmbedder):
    """
    Graph Attention Network for glycan embedding.

    Parameters:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        num_unit (int): number of monosaccharide units
        edge_input_dim (int, optional): dimension of edge features
        num_heads (int, optional): number of attention heads
        negative_slope (float, optional): negative slope of leaky relu activation
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        readout (str, optional): readout function
    """

    def __init__(self, input_dim, hidden_dims, num_unit, edge_input_dim=None, num_heads=4,
                 negative_slope=0.2, short_cut=False, batch_norm=False, activation="relu",
                 concat_hidden=False, readout="mean"):
        super().__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]

        self.input_dim = input_dim
        self.dims = [input_dim] + list(hidden_dims)
        self.num_heads = num_heads
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

        # Embedding
        self.embedding = nn.Embedding(num_unit, input_dim)

        # GAT layers
        self.attention_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None

        for i in range(len(self.dims) - 1):
            self.attention_layers.append(
                nn.MultiheadAttention(self.dims[i], num_heads, batch_first=True)
            )
            self.linear_layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(self.dims[i + 1]))

        # Activation
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        # Readout
        if readout == "sum":
            self.readout = SumReadout()
        elif readout == "mean":
            self.readout = MeanReadout()
        elif readout == "max":
            self.readout = MaxReadout()
        elif readout == "dual":
            self.readout = MeanReadout()
            self.readout_ext = MaxReadout()

        # Output dimensions
        self.output_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1)
        if readout == "dual":
            self.output_dim = self.output_dim * 2
        self.node_output_dim = self.output_dim if readout != "dual" else self.output_dim // 2

    def forward(self, graph, input=None, all_loss=None, metric=None):
        if input is None:
            input = self.embedding(graph.unit_type)

        # Process each graph separately for attention
        batch_size = graph.batch_size
        node2graph = graph.node2graph
        hiddens = []

        for graph_idx in range(batch_size):
            mask = node2graph == graph_idx
            graph_input = input[mask].unsqueeze(0)  # [1, num_nodes, dim]

            layer_input = graph_input
            graph_hiddens = []

            for i, (attn_layer, linear_layer) in enumerate(zip(self.attention_layers, self.linear_layers)):
                # Apply attention
                attn_output, _ = attn_layer(layer_input, layer_input, layer_input)
                attn_output = attn_output + layer_input  # Residual connection

                # Apply linear transformation
                hidden = linear_layer(attn_output)

                if self.batch_norms is not None:
                    # Reshape for batch norm
                    hidden = hidden.squeeze(0)
                    hidden = self.batch_norms[i](hidden)
                    hidden = hidden.unsqueeze(0)

                if i < len(self.linear_layers) - 1:
                    hidden = self.activation(hidden)

                if self.short_cut and hidden.shape == layer_input.shape:
                    hidden = hidden + layer_input

                graph_hiddens.append(hidden.squeeze(0))
                layer_input = hidden

            if self.concat_hidden:
                hiddens.append(torch.cat(graph_hiddens, dim=-1))
            else:
                hiddens.append(graph_hiddens[-1])

        # Combine all nodes back
        node_feature = torch.cat(hiddens, dim=0)

        # Readout
        if hasattr(self, "readout_ext"):
            graph_feature = torch.cat([
                self.readout(graph, node_feature),
                self.readout_ext(graph, node_feature)
            ], dim=-1)
        else:
            graph_feature = self.readout(graph, node_feature)

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature
        }


class GlycanGIN(BaseGlycanEmbedder):
    """
    Graph Isomorphism Network for glycan embedding.

    Parameters:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        num_unit (int): number of monosaccharide units
        edge_input_dim (int, optional): dimension of edge features
        num_mlp_layer (int, optional): number of MLP layers
        eps (float, optional): initial epsilon
        learn_eps (bool, optional): learn epsilon or not
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        readout (str, optional): readout function
    """

    def __init__(self, input_dim=128, hidden_dims=[128, 128, 128], num_unit=143, edge_input_dim=None, num_mlp_layer=2,
                 eps=0, learn_eps=False, short_cut=False, batch_norm=False, activation="relu",
                 concat_hidden=False, readout="sum"):
        super().__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]

        self.input_dim = input_dim
        self.dims = [input_dim] + list(hidden_dims)
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

        # Embedding
        self.embedding = nn.Embedding(num_unit, input_dim)

        # GIN layers
        self.mlps = nn.ModuleList()
        self.eps = nn.ParameterList()

        for i in range(len(self.dims) - 1):
            layers = []
            in_dim = self.dims[i]

            for j in range(num_mlp_layer):
                out_dim = self.dims[i + 1] if j == num_mlp_layer - 1 else self.dims[i]
                layers.append(nn.Linear(in_dim, out_dim))
                if batch_norm and j < num_mlp_layer - 1:
                    layers.append(nn.BatchNorm1d(out_dim))
                if j < num_mlp_layer - 1:
                    if isinstance(activation, str):
                        # Map common activation function names to correct PyTorch names
                        activation_name = activation_map.get(activation.lower(), activation)
                        layers.append(getattr(nn, activation_name)())
                    else:
                        layers.append(activation)
                in_dim = out_dim

            self.mlps.append(nn.Sequential(*layers))

            if learn_eps:
                self.eps.append(nn.Parameter(torch.tensor(eps, dtype=torch.float)))
            else:
                self.eps.append(torch.tensor(eps, dtype=torch.float))

        # Readout
        if readout == "sum":
            self.readout = SumReadout()
        elif readout == "mean":
            self.readout = MeanReadout()
        elif readout == "max":
            self.readout = MaxReadout()
        elif readout == "dual":
            self.readout = MeanReadout()
            self.readout_ext = MaxReadout()

        # Output dimensions
        self.output_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1)
        if readout == "dual":
            self.output_dim = self.output_dim * 2
        self.node_output_dim = self.output_dim if readout != "dual" else self.output_dim // 2

    def forward(self, graph, input=None, all_loss=None, metric=None):
        if input is None:
            input = self.embedding(graph.unit_type)

        hiddens = []
        layer_input = input

        for i, (mlp, epsilon) in enumerate(zip(self.mlps, self.eps)):
            # GIN aggregation (simplified - using mean as aggregation)
            aggregated = scatter_mean(layer_input, graph.node2graph, dim=0, dim_size=graph.batch_size)
            aggregated = aggregated[graph.node2graph]

            # GIN update
            hidden = mlp((1 + epsilon) * layer_input + aggregated)

            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input

            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]

        if hasattr(self, "readout_ext"):
            graph_feature = torch.cat([
                self.readout(graph, node_feature),
                self.readout_ext(graph, node_feature)
            ], dim=-1)
        else:
            graph_feature = self.readout(graph, node_feature)

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature
        }


class GlycanCompGCN(BaseGlycanEmbedder):
    """
    Compositional Graph Convolutional Network for glycan embedding.

    Parameters:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        num_unit (int): number of monosaccharide units
        num_relation (int): number of relations
        edge_input_dim (int, optional): dimension of edge features
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        readout (str, optional): readout function
        composition (str, optional): composition method
    """

    def __init__(self, input_dim, hidden_dims, num_unit, num_relation, edge_input_dim=None,
                 short_cut=False, batch_norm=False, activation="relu", concat_hidden=False,
                 readout="mean", composition="multiply"):
        super().__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]

        self.input_dim = input_dim
        self.dims = [input_dim] + list(hidden_dims)
        self.num_relation = num_relation
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.composition = composition

        # Embeddings
        self.embedding_init = nn.Embedding(num_unit, input_dim)
        self.relation_embedding = nn.Embedding(num_relation, input_dim)

        # CompGCN layers
        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))

        # Activation
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        # Readout
        if readout == "sum":
            self.readout = SumReadout()
        elif readout == "mean":
            self.readout = MeanReadout()
        elif readout == "max":
            self.readout = MaxReadout()
        elif readout == "dual":
            self.readout1 = MeanReadout()
            self.readout2 = MaxReadout()

        # Output dimensions
        self.output_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1)
        if readout == "dual":
            self.output_dim = self.output_dim * 2
        self.node_output_dim = self.output_dim if readout != "dual" else self.output_dim // 2

    def forward(self, graph, input=None, all_loss=None, metric=None):
        hiddens = []
        layer_input = self.embedding_init(graph.unit_type)

        for i, layer in enumerate(self.layers):
            hidden = layer(layer_input)

            if i < len(self.layers) - 1:
                hidden = self.activation(hidden)

            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input

            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]

        if hasattr(self, "readout1"):
            graph_feature = torch.cat([
                self.readout1(graph, node_feature),
                self.readout2(graph, node_feature)
            ], dim=-1)
        else:
            graph_feature = self.readout(graph, node_feature)

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature
        }


class GlycanMPNN(BaseGlycanEmbedder):
    """
    Message Passing Neural Network for glycan embedding.

    Parameters:
        input_dim (int): input dimension
        hidden_dim (int): hidden dimension
        num_unit (int): number of monosaccharide units
        edge_input_dim (int): dimension of edge features
        num_layer (int, optional): number of hidden layers
        num_gru_layer (int, optional): number of GRU layers in each node update
        num_mlp_layer (int, optional): number of MLP layers in each message function
        num_s2s_step (int, optional): number of processing steps in set2set
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output
    """

    def __init__(self, input_dim=143, hidden_dim=128, num_unit=143, edge_input_dim=84, num_layer=3,
                 num_gru_layer=1, num_mlp_layer=2, num_s2s_step=3, short_cut=False,
                 batch_norm=False, activation="relu", concat_hidden=True):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

        # Embedding
        self.embedding_init = nn.Embedding(num_unit, hidden_dim)

        # Message function
        message_layers = []
        for i in range(num_mlp_layer):
            if i == 0:
                message_layers.append(nn.Linear(hidden_dim * 2, hidden_dim))
            else:
                message_layers.append(nn.Linear(hidden_dim, hidden_dim))

            if i < num_mlp_layer - 1:
                if isinstance(activation, str):
                    # Map common activation function names to correct PyTorch names
                    activation_name = activation_map.get(activation.lower(), activation)
                    message_layers.append(getattr(nn, activation_name)())
                else:
                    message_layers.append(activation)

        self.message_func = nn.Sequential(*message_layers)

        # Update function
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_gru_layer)

        # Set2Set readout
        from readout import Set2Set
        feature_dim = hidden_dim * num_layer if concat_hidden else hidden_dim
        self.readout = Set2Set(feature_dim, num_step=num_s2s_step)

        # Output dimensions
        self.output_dim = feature_dim * 2  # Set2Set doubles the dimension
        self.node_output_dim = feature_dim

    def forward(self, graph, input=None, all_loss=None, metric=None):
        hiddens = []
        layer_input = self.embedding_init(graph.unit_type)
        hx = layer_input.repeat(self.gru.num_layers, 1, 1)

        for i in range(self.num_layer):
            # Message passing (simplified)
            # In practice, this should aggregate messages from neighbors
            messages = self.message_func(
                torch.cat([layer_input, layer_input], dim=-1)
            )

            # Update
            hidden, hx = self.gru(messages.unsqueeze(0), hx)
            hidden = hidden.squeeze(0)

            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input

            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]

        graph_feature = self.readout(graph, node_feature)

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature
        }


# ============================================================================
# Sequence-based Embedders
# ============================================================================

class GlycanConvolutionalNetwork(BaseGlycanEmbedder):
    """
    Convolutional Neural Network for glycan embedding.

    Parameters:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        glycoword_dim (int): number of glycowords
        kernel_size (int, optional): size of convolutional kernel
        stride (int, optional): stride of convolution
        padding (int, optional): padding added to both sides of the input
        activation (str or function, optional): activation function
        short_cut (bool, optional): use short cut or not
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        readout (str, optional): readout function
        max_length (int, optional): maximum sequence length
    """

    def __init__(self, input_dim, hidden_dims, glycoword_dim, kernel_size=3, stride=1,
                 padding=1, activation='relu', short_cut=False, concat_hidden=False,
                 readout="max", max_length=512):
        super().__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]

        self.input_dim = input_dim
        self.dims = [input_dim] + list(hidden_dims)
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.padding_id = input_dim - 1
        self.max_length = max_length

        # Embedding
        self.embedding_init = nn.Embedding(glycoword_dim, input_dim)

        # Activation
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        # Convolutional layers
        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            # Calculate proper padding to preserve sequence length
            # For stride=1, padding = (kernel_size - 1) // 2 preserves length
            proper_padding = (kernel_size - 1) // 2 if stride == 1 else padding
            self.layers.append(
                nn.Conv1d(self.dims[i], self.dims[i + 1], kernel_size, stride, proper_padding)
            )

        # Readout
        if readout == "sum":
            self.readout = SumReadout('glycan')
        elif readout == "mean":
            self.readout = MeanReadout('glycan')
        elif readout == "max":
            self.readout = MaxReadout('glycan')
        elif readout == "attention":
            output_dim = sum(hidden_dims) if concat_hidden else hidden_dims[-1]
            self.readout = AttentionReadout(output_dim, 'glycan')

        # Output dimension
        self.output_dim = sum(hidden_dims) if concat_hidden else hidden_dims[-1]
        self.node_output_dim = self.output_dim

    def forward(self, graph, input=None, all_loss=None, metric=None):
        # Get glycoword sequences
        input = graph.glycoword_type.long()
        input = self.embedding_init(input)

        # Convert to padded format for convolution
        input, mask = variadic_to_padded(input, graph.num_glycowords, self.padding_id)

        hiddens = []
        layer_input = input

        for layer in self.layers:
            # Conv1d expects (batch, channels, length)
            hidden = layer(layer_input.transpose(1, 2)).transpose(1, 2)
            hidden = self.activation(hidden)

            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input

            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            # Check if all hidden tensors have the same sequence length
            if len(set(h.size(1) for h in hiddens)) == 1:
                hidden = torch.cat(hiddens, dim=-1)
            else:
                # If sequence lengths differ, just use the last layer
                hidden = hiddens[-1]
        else:
            hidden = hiddens[-1]

        # Convert back to variadic format
        glycoword_feature = padded_to_variadic(hidden, graph.num_glycowords)
        graph_feature = self.readout(graph, glycoword_feature)

        return {
            "graph_feature": graph_feature,
            "glycoword_feature": glycoword_feature
        }


class GlycanResNet(BaseGlycanEmbedder):
    """
    Residual Network for glycan embedding.

    Parameters:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        glycoword_dim (int): number of glycowords
        num_blocks (int, optional): number of residual blocks
        kernel_size (int, optional): size of convolutional kernel
        activation (str or function, optional): activation function
        short_cut (bool, optional): use short cut or not
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        layer_norm (bool, optional): apply layer normalization or not
        dropout (float, optional): dropout ratio of input features
        readout (str, optional): readout function
        max_length (int, optional): maximum sequence length
    """

    def __init__(self, input_dim=512, hidden_dims=[512, 512, 512, 512, 512, 512, 512, 512],
                 glycoword_dim=216, num_blocks=3, kernel_size=3,
                 activation="gelu", short_cut=True, concat_hidden=False, layer_norm=True,
                 dropout=0.1, readout="mean", max_length=512):
        super().__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]

        self.input_dim = input_dim
        self.dims = list(hidden_dims)
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.padding_id = input_dim - 1
        self.max_length = max_length

        # Embeddings
        self.embedding_init = nn.Embedding(glycoword_dim, input_dim)
        self.embedding = nn.Linear(input_dim, hidden_dims[0])
        self.position_embedding = nn.Embedding(max_length, hidden_dims[0])

        # Layer norm and dropout
        if layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dims[0])
        else:
            self.layer_norm = None

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        # Activation
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        # Residual blocks
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            block = nn.Sequential(
                nn.Conv1d(hidden_dims[0], hidden_dims[0], kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(hidden_dims[0]),  # Use BatchNorm1d which works with Conv1d
                nn.ReLU() if activation == "relu" else nn.GELU(),
                nn.Conv1d(hidden_dims[0], hidden_dims[0], kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(hidden_dims[0])
            )
            self.blocks.append(block)

        # Readout
        if readout == "sum":
            self.readout = SumReadout("glycan")
        elif readout == "mean":
            self.readout = MeanReadout("glycan")
        elif readout == "attention":
            self.readout = AttentionReadout(hidden_dims[0], "glycan")

        # Output dimension
        self.output_dim = hidden_dims[0]
        self.node_output_dim = hidden_dims[0]

    def forward(self, graph, input=None, all_loss=None, metric=None):
        input = graph.glycoword_type.long()
        input = self.embedding_init(input)
        input, mask = variadic_to_padded(input, graph.num_glycowords, self.padding_id)
        mask = mask.unsqueeze(-1)

        # Apply embeddings
        input = self.embedding(input)
        positions = torch.arange(input.size(1), device=input.device)
        input = input + self.position_embedding(positions).unsqueeze(0)

        if self.layer_norm:
            input = self.layer_norm(input)
        if self.dropout:
            input = self.dropout(input)

        input = input * mask

        # Apply residual blocks
        hidden = input
        for block in self.blocks:
            # Conv1d expects (batch, channels, length)
            residual = hidden
            hidden = hidden.transpose(1, 2)
            hidden = block(hidden)
            hidden = hidden.transpose(1, 2)
            hidden = self.activation(hidden + residual) * mask

        glycoword_feature = padded_to_variadic(hidden, graph.num_glycowords)
        graph_feature = self.readout(graph, glycoword_feature)

        return {
            "graph_feature": graph_feature,
            "glycoword_feature": glycoword_feature
        }


class GlycanLSTM(BaseGlycanEmbedder):
    """
    Long Short-Term Memory network for glycan embedding.

    Parameters:
        input_dim (int): input dimension
        hidden_dim (int): hidden dimension
        glycoword_dim (int): number of glycowords
        num_layers (int): number of LSTM layers
        activation (str or function, optional): activation function
        layer_norm (bool, optional): apply layer normalization or not
        dropout (float, optional): dropout ratio
        bidirectional (bool, optional): use bidirectional LSTM
        max_length (int, optional): maximum sequence length
    """

    def __init__(self, input_dim=21, hidden_dim=640, glycoword_dim=216, num_layers=2, activation='tanh',
                 layer_norm=False, dropout=0, bidirectional=True, max_length=512):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.padding_id = input_dim - 1
        self.max_length = max_length
        self.bidirectional = bidirectional

        # Embeddings
        self.embedding_init = nn.Embedding(glycoword_dim, input_dim)
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # Layer norm and dropout
        if layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)
        else:
            self.layer_norm = None

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        # Activation
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        # LSTM
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0, bidirectional=bidirectional)

        # Output projection
        self.reweight = nn.Linear(2 * num_layers if bidirectional else num_layers, 1)
        self.linear = nn.Linear(hidden_dim, hidden_dim)

        # Output dimensions
        self.output_dim = hidden_dim
        self.node_output_dim = 2 * hidden_dim if bidirectional else hidden_dim

    def forward(self, graph, input=None, all_loss=None, metric=None):
        input = graph.glycoword_type.long()
        input = self.embedding_init(input)
        input, _ = variadic_to_padded(input, graph.num_glycowords, self.padding_id)

        input = self.embedding(input)
        if self.layer_norm:
            input = self.layer_norm(input)
        if self.dropout:
            input = self.dropout(input)

        output, (hidden, _) = self.lstm(input)

        # Convert output back to variadic
        glycoword_feature = padded_to_variadic(output, graph.num_glycowords)

        # Process hidden states for graph representation
        # hidden shape: (num_layers * num_directions, batch, hidden_dim)
        if self.bidirectional:
            # Separate forward and backward
            hidden = hidden.view(self.num_layers, 2, graph.batch_size, self.hidden_dim)
            # Concatenate forward and backward final states from each layer
            hidden = hidden.permute(2, 0, 1, 3)  # (batch, num_layers, 2, hidden_dim)
            hidden = hidden.reshape(graph.batch_size, self.num_layers * 2, self.hidden_dim)  # (batch, 2*num_layers, hidden_dim)
        else:
            # For unidirectional: (num_layers, batch, hidden_dim) -> (batch, num_layers, hidden_dim)
            hidden = hidden.permute(1, 0, 2)

        # Apply reweight to get a weighted combination of layer outputs
        # hidden shape: (batch, num_layers[*2], hidden_dim)
        # reweight expects: (batch, num_layers[*2])
        # So we need to aggregate across the hidden dimension first
        hidden_pooled = hidden.mean(dim=-1)  # (batch, num_layers[*2])
        layer_weights = self.reweight(hidden_pooled)  # (batch, 1)
        layer_weights = torch.softmax(layer_weights, dim=0)  # Normalize across batch
        
        # Get weighted sum of all layers' final hidden states
        graph_feature = (hidden * layer_weights.unsqueeze(-1)).sum(dim=1)  # (batch, hidden_dim)
        graph_feature = self.linear(graph_feature)
        graph_feature = self.activation(graph_feature)

        return {
            "graph_feature": graph_feature,
            "glycoword_feature": glycoword_feature
        }


class GlycanBERT(BaseGlycanEmbedder):
    """
    Bidirectional Encoder Representations from Transformers for glycan embedding.

    Parameters:
        input_dim (int): input dimension (vocabulary size)
        hidden_dim (int): hidden dimension
        num_layers (int): number of Transformer blocks
        num_heads (int): number of attention heads
        intermediate_dim (int): intermediate hidden dimension of Transformer block
        activation (str or function, optional): activation function
        hidden_dropout (float, optional): dropout ratio of hidden features
        attention_dropout (float, optional): dropout ratio of attention maps
        max_position (int, optional): maximum number of positions
    """

    def __init__(self, input_dim=216, hidden_dim=768, num_layers=8, num_heads=12,
                 intermediate_dim=3072, activation="gelu", hidden_dropout=0.1,
                 attention_dropout=0.1, max_position=8192):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_position = max_position

        # Special tokens
        self.num_glycoword_type = input_dim

        # Embeddings
        self.embedding = nn.Embedding(input_dim + 3, hidden_dim)  # +3 for BOS, EOS, PAD
        self.position_embedding = nn.Embedding(max_position, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(hidden_dropout)

        # Transformer layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=intermediate_dim,
                dropout=hidden_dropout,
                activation=activation,
                batch_first=True
            )
            self.layers.append(layer)

        # Output projection
        self.linear = nn.Linear(hidden_dim, hidden_dim)

        # Output dimensions
        self.output_dim = hidden_dim
        self.node_output_dim = hidden_dim

    def forward(self, graph, input=None, all_loss=None, metric=None):
        input = graph.glycoword_type
        size_ext = graph.num_glycowords.clone()

        # Prepare special tokens
        batch_size = graph.batch_size
        device = input.device

        # Add BOS token
        bos = torch.ones(batch_size, dtype=torch.long, device=device) * self.num_glycoword_type
        bos_sizes = torch.ones_like(size_ext)

        # Manually extend with BOS
        bos_extended = []
        input_offset = 0
        for i, size in enumerate(graph.num_glycowords):
            bos_extended.append(bos[i:i + 1])
            bos_extended.append(input[input_offset:input_offset + size])
            input_offset += size
        input = torch.cat(bos_extended)
        size_ext = size_ext + bos_sizes

        # Add EOS token
        eos = torch.ones(batch_size, dtype=torch.long, device=device) * (self.num_glycoword_type + 1)
        eos_extended = []
        input_offset = 0
        for i, size in enumerate(size_ext):
            eos_extended.append(input[input_offset:input_offset + size])
            eos_extended.append(eos[i:i + 1])
            input_offset += size
        input = torch.cat(eos_extended)
        size_ext = size_ext + torch.ones_like(size_ext)

        # Convert to padded format
        input, mask = variadic_to_padded(input, size_ext, self.num_glycoword_type + 2)
        mask = mask.float()

        # Apply embeddings
        input = self.embedding(input)
        position_indices = torch.arange(input.shape[1], device=device)
        input = input + self.position_embedding(position_indices).unsqueeze(0)
        input = self.layer_norm(input)
        input = self.dropout(input)

        # Apply transformer layers
        for layer in self.layers:
            # Create attention mask (1 for positions to mask)
            attn_mask = (1.0 - mask) * -10000.0
            input = layer(input, src_key_padding_mask=attn_mask.squeeze(-1) < -1)

        # Get glycoword features (excluding special tokens)
        glycoword_feature = padded_to_variadic(input, size_ext)

        # Get graph feature from [CLS] token (first position)
        graph_feature = input[:, 0]
        graph_feature = self.linear(graph_feature)
        graph_feature = torch.tanh(graph_feature)

        return {
            "graph_feature": graph_feature,
            "glycoword_feature": glycoword_feature
        }


# ============================================================================
# Main Embedder Class
# ============================================================================

class GlycanEmbedder:
    """
    Unified interface for all glycan embedding methods.
    """

    EMBEDDER_CLASSES = {
        # Graph-based
        'gcn': GlycanGCN,
        'rgcn': GlycanRGCN,
        'gat': GlycanGAT,
        'gin': GlycanGIN,
        'compgcn': GlycanCompGCN,
        'mpnn': GlycanMPNN,
        # Sequence-based
        'cnn': GlycanConvolutionalNetwork,
        'resnet': GlycanResNet,
        'lstm': GlycanLSTM,
        'bert': GlycanBERT
    }

    GRAPH_METHODS = ['gcn', 'rgcn', 'gat', 'gin', 'compgcn', 'mpnn']
    SEQUENCE_METHODS = ['cnn', 'resnet', 'lstm', 'bert']
    ALL_METHODS = GRAPH_METHODS + SEQUENCE_METHODS

    def __init__(self, vocab_path: Optional[str] = None, device: str = 'cpu'):
        self.device = torch.device(device)
        self._load_vocabulary(vocab_path)
        self._embedders = {}

    def _load_vocabulary(self, vocab_path: Optional[str]):
        """Load glycan vocabulary mappings."""
        if vocab_path and os.path.exists(vocab_path):
            with open(vocab_path, 'rb') as f:
                entities = pkl.load(f)
        else:
            print("Warning: Using minimal fallback vocabulary.")
            entities = self._create_minimal_vocab()

        # Separate units and links
        self.units = [entity for entity in entities
                      if not (entity.startswith(("a", "b", "?")) or re.match("^[0-9]+(-[0-9]+)+$", entity))]
        self.unit2id = {x: i for i, x in enumerate(self.units)}

        self.links = [entity for entity in entities
                      if entity.startswith(("a", "b", "?")) or re.match("^[0-9]+(-[0-9]+)+$", entity)]
        self.link2id = {x: i for i, x in enumerate(self.links)}

        # Glycowords
        self.glycowords = entities + ["[", "]", "{", "}", "Unknown_Token"]
        self.glycoword2id = {x: i for i, x in enumerate(self.glycowords)}

        print(f"Loaded vocabulary: {len(self.units)} units, {len(self.links)} links, {len(self.glycowords)} glycowords")

    def _create_minimal_vocab(self) -> List[str]:
        """Create minimal vocabulary for basic functionality."""
        units = ["Glc", "Gal", "Man", "GlcNAc", "GalNAc", "Fuc", "Xyl", "Neu5Ac", "Neu5Gc"]
        links = ["a1-2", "a1-3", "a1-4", "a1-6", "b1-2", "b1-3", "b1-4", "b1-6"]
        return units + links

    def _parse_glycan_to_units_and_links(self, iupac: str) -> tuple:
        """Parse IUPAC string to extract units and linkage information."""
        if not GLYCOWORK_AVAILABLE:
            return self._simple_parse(iupac)

        # Use glycowork for robust parsing
        units_links = [x for x in self._multireplace(iupac,
                                                     {"[": "", "]": "", "{": "", "}": "", ")": "("}).split("(") if x]

        units = [x for x in units_links
                 if not (x.startswith(("a", "b", "?")) or re.match("^[0-9]+(-[0-9]+)+$", x))]

        # Convert to IDs
        unit_ids = []
        for unit in units:
            core_unit = tokenization.get_core(unit) if GLYCOWORK_AVAILABLE else unit
            unit_ids.append(self.unit2id.get(core_unit, len(self.units) - 1))

        # Extract glycowords
        glycoword_ids = []
        for unit_link in units_links:
            core = tokenization.get_core(unit_link) if GLYCOWORK_AVAILABLE else unit_link
            glycoword_ids.append(self.glycoword2id.get(core, len(self.glycowords) - 1))

        return torch.tensor(unit_ids), torch.tensor(glycoword_ids)

    def _simple_parse(self, iupac: str) -> tuple:
        """Simple parsing when glycowork is not available."""
        unit_ids = []
        glycoword_ids = []

        for unit in self.units:
            if unit in iupac:
                unit_ids.append(self.unit2id[unit])
                glycoword_ids.append(self.glycoword2id.get(unit, len(self.glycowords) - 1))

        if not unit_ids:
            unit_ids = [len(self.units) - 1] if self.units else [0]
            glycoword_ids = [len(self.glycowords) - 1]

        return torch.tensor(unit_ids), torch.tensor(glycoword_ids)

    @staticmethod
    def _multireplace(string: str, replace_dict: Dict[str, str]) -> str:
        """Replace multiple substrings in a string."""
        for k, v in replace_dict.items():
            string = string.replace(k, v)
        return string

    def _create_embedder(self, method: str, embedding_dim: int, input_dim: int, **kwargs):
        """Create embedder for specified method."""
        method = method.lower()
        if method not in self.ALL_METHODS:
            raise ValueError(f"Unknown method '{method}'. Available: {self.ALL_METHODS}")

        embedder_class = self.EMBEDDER_CLASSES[method]

        # Set default parameters based on method type
        if method in self.GRAPH_METHODS:
            num_units = len(self.units)
            params = {
                'input_dim': input_dim,
                'hidden_dims': [embedding_dim],
                'num_unit': num_units
            }

            if method in ['rgcn', 'compgcn']:
                params['num_relation'] = len(self.links)
            elif method == 'mpnn':
                params['hidden_dim'] = embedding_dim
                params.pop('hidden_dims')

        else:  # sequence methods
            if method == 'bert':
                # BERT uses input_dim as vocabulary size, not glycoword_dim
                # Ensure hidden_dim is divisible by num_heads
                # Find the largest divisor of embedding_dim that's <= 12
                valid_heads = [h for h in [1, 2, 3, 4, 6, 8, 12] if embedding_dim % h == 0]
                num_heads = max(valid_heads) if valid_heads else 1
                
                params = {
                    'input_dim': len(self.glycowords),
                    'hidden_dim': embedding_dim,
                    'num_heads': num_heads
                }
            else:
                # Other sequence methods (CNN, ResNet, LSTM)
                params = {
                    'input_dim': input_dim,
                    'hidden_dims': [embedding_dim],
                    'glycoword_dim': len(self.glycowords)
                }

                if method == 'lstm':
                    params['hidden_dim'] = embedding_dim
                    params.pop('hidden_dims')
                    params['num_layers'] = kwargs.pop('num_layers', 2)

        # Update with user kwargs
        params.update(kwargs)

        return embedder_class(**params).to(self.device)

    def embed_glycans(self, glycan_list: List[str], method: str = 'gcn',
                      embedding_dim: int = 128, input_dim: int = None, **kwargs) -> torch.Tensor:
        """
        Embed glycans using specified method.
        """
        method = method.lower()

        # Set default input_dim
        if input_dim is None:
            if method in self.GRAPH_METHODS:
                input_dim = 128
            else:
                if method == 'bert':
                    input_dim = len(self.glycowords)
                else:
                    input_dim = 1024

        # Create embedder if not exists
        embedder_key = f"{method}_{input_dim}_{embedding_dim}"
        if embedder_key not in self._embedders:
            self._embedders[embedder_key] = self._create_embedder(
                method, embedding_dim, input_dim, **kwargs
            )

        embedder = self._embedders[embedder_key]

        # Parse glycans and create graph structure
        unit_ids_batch = []
        glycoword_ids_batch = []

        for glycan in glycan_list:
            unit_ids, glycoword_ids = self._parse_glycan_to_units_and_links(glycan)
            unit_ids_batch.append(unit_ids)
            glycoword_ids_batch.append(glycoword_ids)

        # Create mock graph object
        graph = self._create_graph_object(unit_ids_batch, glycoword_ids_batch, method)

        # Get embeddings
        with torch.no_grad():
            output = embedder(graph)
            embeddings = output["graph_feature"]

        return embeddings

    def _create_graph_object(self, unit_ids_batch, glycoword_ids_batch, method):
        """Create a mock graph object for the embedders."""

        class MockGraph:
            pass

        graph = MockGraph()
        graph.batch_size = len(unit_ids_batch)

        if method in self.GRAPH_METHODS:
            # For graph methods
            graph.unit_type = torch.cat([ids.to(self.device) for ids in unit_ids_batch])

            # Create node2graph mapping
            node2graph = []
            for i, ids in enumerate(unit_ids_batch):
                node2graph.extend([i] * len(ids))
            graph.node2graph = torch.tensor(node2graph, device=self.device)

        else:
            # For sequence methods
            graph.glycoword_type = torch.cat([ids.to(self.device) for ids in glycoword_ids_batch])
            graph.num_glycowords = torch.tensor([len(ids) for ids in glycoword_ids_batch], device=self.device)

            # Create glycoword2graph mapping
            glycoword2graph = []
            for i, ids in enumerate(glycoword_ids_batch):
                glycoword2graph.extend([i] * len(ids))
            graph.glycoword2graph = torch.tensor(glycoword2graph, device=self.device)

        return graph

    def get_available_methods(self) -> Dict[str, List[str]]:
        """Get all available embedding methods."""
        return {
            'graph_based': self.GRAPH_METHODS,
            'sequence_based': self.SEQUENCE_METHODS,
            'all': self.ALL_METHODS
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def embed_glycans(glycan_list: List[str], method: str = 'gcn',
                  embedding_dim: int = 128, input_dim: int = None,
                  vocab_path: Optional[str] = None, device: str = 'cpu',
                  **kwargs) -> np.ndarray:
    """Quick function to embed glycans with any method."""
    embedder = GlycanEmbedder(vocab_path, device)
    embeddings = embedder.embed_glycans(glycan_list, method, embedding_dim, input_dim, **kwargs)
    return embeddings.cpu().numpy()


def get_available_methods() -> Dict[str, List[str]]:
    """Get all available embedding methods."""
    return {
        'graph_based': GlycanEmbedder.GRAPH_METHODS,
        'sequence_based': GlycanEmbedder.SEQUENCE_METHODS,
        'all': GlycanEmbedder.ALL_METHODS
    }


if __name__ == "__main__":
    # Example usage
    glycans = [
        "Gal(a1-3)[Fuc(a1-2)]Gal(b1-4)GlcNAc",
        "Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc",
        "Neu5Ac(a2-3)Gal(b1-4)Glc"
    ]

    print("Available methods:")
    for category, methods in get_available_methods().items():
        print(f"  {category}: {methods}")

    # Test different embedding methods
    for method in ['gcn', 'lstm', 'cnn', 'bert', 'rgcn', 'compgcn', 'mpnn']:
        embeds = embed_glycans(glycans, method=method, embedding_dim=512,
                               vocab_path="glycoword_vocab.pkl", device="cuda")
        print(f"{method.upper()} embeddings shape: {embeds.shape}")