import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from matplotlib.colors import LogNorm
from PIL import Image


def show_artifacts(
    test_model: nn.Module,
    test_image: torch.Tensor,
    log_scale=False,
    token: int = 0,
    shape: tuple = (24, 24),
    discard_tokens: int = 0,
    num_cols: int = 4,
    unfrozen_layers: int = 0,
) -> None:
    """
    Generate the Attention maps and the norm values for the DEIT-III model

    test_model: DEIT-III model to be tested,
    test_image: Image of the correct size for the corresponding model, and batch dimension is accounted for
    log_scale: If True, the log of the attention map values will be displayed
    token: The token to be visualized in the attention maps
    """

    test_model(test_image)
    num_blocks = len(test_model.blocks)

    ## 1. Norm of feature values after MLP
    print("Norm of feature values after MLP")
    output = test_model.block_output[f"block{num_blocks-1}"]
    output = output.squeeze(0)
    # output = output[1:] # !!!!!!!!!!!
    # copmute norm of all output elements
    output_norms = output.norm(dim=-1)
    # output_norms.shape

    # TODO: double check
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # discard_tokens = 4 # discard the CLS token and 4 register tokens
    if discard_tokens > 0:
        cls = output_norms[0]
        registers = output_norms[-discard_tokens:]
        output_norms = output_norms[1:-discard_tokens]
    else:
        cls = output_norms[0]
        output_norms = output_norms[1:]
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    plt.imshow(output_norms.reshape(shape[0], shape[1]).detach().numpy())
    plt.axis("off")
    plt.colorbar(label="Norm Values")  # add a colorbar as a legend
    plt.show()

    if discard_tokens > 0:
        print(
            "Norm of register tokens: ",
            [round(float(x), 3) for x in registers.detach().numpy()],
        )
    print("Norm of CLS token: ", (cls.detach().numpy()))

    plt.hist(output_norms.detach().numpy(), bins=50)
    plt.xlabel("Norm Values")
    plt.ylabel("Frequency")
    plt.show()

    #########################################################################################################

    ## 2. Attention maps for the last Attention Head
    # print("Attention maps for the last Attention Head")
    # attn_map_mean = (
    #     test_model.blocks[num_blocks - 1].attn.attn_map.squeeze(0).mean(dim=0)
    # )
    # ### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # if discard_tokens > 0:
    #     attn_map_mean = attn_map_mean[token][1:-discard_tokens]
    # else:
    #     attn_map_mean = attn_map_mean[token][1:]
    # if log_scale:
    #     attn_map_mean = torch.log(attn_map_mean + 1e-6)
    # # attn_map_mean.shape

    # plt.imshow(attn_map_mean.reshape(shape[0], shape[1]).detach().numpy())
    # plt.axis("off")
    # plt.colorbar(label="CLS attention map")
    # plt.show()
    #########################################################################################################

    ## 3. All attention maps
    print("All attention maps")

    num_rows = (
        num_blocks + num_cols - 1
    ) // num_cols  # calculate the number of rows needed
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 2.5))
    axes = axes.flatten()

    attn_reg = []

    for i in range(num_blocks):
        attn_map = test_model.blocks[i].attn.attn_map.squeeze(0).mean(dim=0)
        # attn_map = attn_map[token][1:] # !!!!!!!!!!!
        if discard_tokens > 0:
            attn_cls = attn_map[token][0]
            attn_reg.append(attn_map[token][-discard_tokens:])
            attn_map = attn_map[token][1:-discard_tokens]
        else:
            attn_cls = attn_map[token][0]
            attn_map = attn_map[token][1:]
        attn_map_img = attn_map.reshape(shape[0], shape[1]).detach().numpy()

        im = axes[i].imshow(attn_map_img)
        axes[i].axis("off")
        axes[i].set_title(f"Block {i+1}, CLS: {attn_cls:.3f}")
        fig.colorbar(im, ax=axes[i], orientation="vertical")

    # Hide any remaining empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

    if discard_tokens > 0:
        for i in list(range(unfrozen_layers))[::-1]:
            print(
                "Attention of register tokens in block",
                num_blocks - i,
                "=",
                [
                    round(float(x), 4)
                    for x in attn_reg[num_blocks - 1 - i].detach().numpy()
                ],
            )
    print("\n")
    #########################################################################################################

    ## 4. All norm maps
    print("All norm maps")

    num_rows = (
        num_blocks + num_cols - 1
    ) // num_cols  # calculate the number of rows needed
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 2.5))
    axes = axes.flatten()
    output_norms_reg = []

    for i in range(num_blocks):
        output = test_model.block_output[f"block{i}"]
        output = output.squeeze(0)
        # output = output[1:] # !!!!!!!!!!!
        if discard_tokens > 0:
            output_cls = output[0]
            output_reg = output[-discard_tokens:]
            output = output[1:-discard_tokens]
            output_norms_reg.append(output_reg.norm(dim=-1))
        else:
            output_cls = output[0]
            output = output[1:]
        output_norms_cls = output_cls.norm(dim=-1)
        
        output_norms = output.norm(dim=-1)
        output_norms_img = output_norms.reshape(shape[0], shape[1]).detach().numpy()

        im = axes[i].imshow(output_norms_img)
        axes[i].axis("off")
        axes[i].set_title(
            f"Block {i+1}, cls = " + str(round(output_norms_cls.item(), 2))
        )
        fig.colorbar(im, ax=axes[i], orientation="vertical")

    # Hide any remaining empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

    # print(output_norms_reg)
    if discard_tokens > 0:
        for i in list(range(unfrozen_layers))[::-1]:
            print(
                "Norm of register tokens in block",
                num_blocks - i,
                "=",
                [
                    round(float(x), 3)
                    for x in output_norms_reg[num_blocks - 1 - i].detach().numpy()
                ],
            )

    #########################################################################################################


def get_image(image_path: str, img_shape: tuple = (384, 384)) -> torch.Tensor:
    """Load an image, resize, normalize, and convert to a tensor with a batch dimension."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(img_shape)
    img = np.array(img).transpose(2, 0, 1)  # convert to numpy and rearrange dimensions
    img = (
        torch.tensor(img, dtype=torch.float32) / 255.0
    )  # normalize and convert to tensor
    return img.unsqueeze(0)  # Add batch dimension


def plot_image(image_tensor: torch.Tensor) -> None:
    """Plot a single image tensor."""
    image_np = (
        image_tensor.permute(1, 2, 0).clip(0, 1).numpy()
    )  # Convert to (H, W, C) and clip values
    plt.imshow(image_np)
    plt.axis("off")
    plt.show()


def plot_feature_norms_with_high_attn(
    chosen_model: nn.Module,
    output_tensor: torch.Tensor,
    high_attn_tokens: torch.Tensor,
    grid_size: tuple = (24, 24),
    dot_color: str = "red",
    dot_size: int = 30,
    label: str = "High Attention Tokens",
    discard_tokens: int = 0,
) -> None:
    """
    Plots the norm of feature values from a model's MLP output, with red dots indicating high attention tokens.

    :param output_tensor: The output tensor from the model's MLP layer
    :param high_attn_tokens: The indices of the high attention tokens
    :param grid_size: The size of the grid to reshape the output tensor
    :param dot_color: The color of the high attention points
    :param dot_size: The size of the high attention points
    """
    # squeeze and remove the CLS token
    if discard_tokens > 0:
        output = output_tensor.squeeze(0)[1:-discard_tokens]
    else:
        output = output_tensor.squeeze(0)[1:]
    # compute norm of all output elements
    output_norms = output.norm(dim=-1)

    # plot the norm values as an image
    plt.subplot(1, 2, 1)
    plt.imshow(output_norms.reshape(grid_size).detach().numpy())
    plt.axis("off")
    plt.colorbar(label="Norm Values")

    # highlight high attention points with red dots
    high_attn_indices = np.unravel_index(high_attn_tokens[0], grid_size)
    plt.scatter(
        high_attn_indices[1],
        high_attn_indices[0],
        color=dot_color,
        s=dot_size,
        label=label,
    )
    plt.legend(loc="upper right")

    # plot the attention map next to it
    plt.subplot(1, 2, 2)
    attn_map = chosen_model.blocks[-1].attn.attn_map.squeeze(0).mean(dim=0)
    if discard_tokens > 0:
        attn_map = attn_map[0][1:-discard_tokens].reshape(grid_size).detach().numpy()
    else:
        attn_map = attn_map[0][1:].reshape(grid_size).detach().numpy()
    plt.imshow(attn_map)
    plt.axis("off")
    plt.colorbar(label="Attention Map")

    plt.gcf().set_size_inches(12, 6)  # Make the image bigger
    # plt.show()set_size_inches(12, 6)  # Make the image bigger
    # plt.show()set_size_inches(12, 6)  # Make the image bigger
    # plt.show()set_size_inches(12, 6)  # Make the image bigger
    plt.show()


def show_attn_progression(
    test_model: nn.Module,
    token: str = "cls",
    grid_size: tuple = (24, 24),
    discard_tokens: int = 0,
    save_path: str = None,
    token_name: str = "CLS",
) -> None:

    ## All attention maps
    num_images = len(test_model.blocks)
    num_cols = 6
    num_rows = (
        num_images + num_cols - 1
    ) // num_cols  # calculate the number of rows needed

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 2.5))
    axes = axes.flatten()

    for i in range(num_images):
        attn_map = test_model.blocks[i].attn.attn_map.squeeze(0).mean(dim=0)
        if str(token) == "cls":
            attn_cls = attn_map[0][0]
            if discard_tokens > 0:
                attn_map = attn_map[0][1:-discard_tokens]
            else:
                attn_map = attn_map[0][1:]

        elif str(token) == "reg":
            for j in range(1, discard_tokens + 1):
                plt.close(fig)
                if save_path is not None:
                    save_path = save_path[:-8] + f"reg{j}.png"
                show_attn_progression(
                    test_model,
                    token=-1 - j,
                    grid_size=grid_size,
                    discard_tokens=discard_tokens,
                    save_path=save_path,
                    token_name=f"Reg{j}",
                )
            return

        else:
            attn_cls = attn_map[token + 1][0]
            if discard_tokens > 0:
                attn_map = attn_map[token + 1][1:-discard_tokens]
            else:
                attn_map = attn_map[token + 1][1:]

        attn_map_img = attn_map.reshape(grid_size).detach().numpy()

        # im = axes[i].imshow(attn_map_img)
        # axes[i].axis("off")
        # axes[i].set_title(f"Block {i+1}, CLS: {attn_cls:.3f}")
        # fig.colorbar(im, ax=axes[i], orientation='vertical')

        im = axes[i].imshow(attn_map_img)
        axes[i].axis("off")
        axes[i].set_title(f"Block {i+1}, {token_name}: {attn_cls:.3f}")
        # fig.colorbar(im, ax=axes[i], orientation="vertical")

    # hide any remaining empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_similarity_density(
    top_norm_similarities: list, not_top_norm_similarities: list
) -> None:
    plt.figure(figsize=(6, 4))
    sns.kdeplot(
        not_top_norm_similarities, label="normal patches", color="blue", bw_adjust=0.5
    )
    sns.kdeplot(
        top_norm_similarities, label="artifact patches", color="orange", bw_adjust=0.5
    )

    plt.xlabel("Cosine similarity")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_norm_proportions(
    layer_norms,
    num_bins_layer: int = 100,
    num_bins_norm: int = 200,
    norm_range: tuple = (3, 300),
    cmap: str = "magma",
) -> None:
    """
    Plots the proportion of norm values across layers with a density heatmap.

    :param layer_norms: list of numpy arrays, each array contains the norm values for a specific layer.
    :param num_bins_layer: number of bins to divide the layers. Default is 100.
    :param num_bins_norm: number of bins to divide the norm values. Default is 200.
    :param norm_range: the min and max range for norm values.
    :param cmap: colormap to use for the heatmap.
    """
    # prepare bins for norms and layers
    all_layers = np.repeat(np.arange(len(layer_norms)), [len(x) for x in layer_norms])
    all_norms = np.hstack(layer_norms)
    layer_edges = np.linspace(0.5, len(layer_norms) + 0.5, num_bins_layer + 1)
    norm_edges = np.logspace(
        np.log10(norm_range[0]), np.log10(norm_range[1]), num_bins_norm + 1
    )

    # calculate histogram and normalize within each layer
    hist, _, _ = np.histogram2d(
        all_layers + 1, all_norms, bins=[layer_edges, norm_edges]
    )
    layer_totals = hist.sum(axis=1, keepdims=True)  # Total counts per layer
    proportions = hist / (layer_totals + 1e-6)  # Avoid division by zero

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.set_facecolor("black")  # set background to black

    # display the heatmap
    plt.imshow(
        proportions.T,
        aspect="auto",
        origin="lower",
        extent=[1, len(layer_norms), norm_edges[0], norm_edges[-1]],
        norm=LogNorm(vmin=1e-5, vmax=1),  # Avoid log(0) errors
        cmap=cmap,
    )

    plt.colorbar(label="Proportion")
    plt.yscale("log")
    plt.xlabel("Layer", fontsize=14)
    plt.ylabel("Norm", fontsize=14)
    plt.title("Proportion of Norms Across Layers", fontsize=16)
    plt.tight_layout()
    plt.show()


def norm_per_layer(chosen_model: nn.Module, discard_tokens: int = 0) -> list:
    layer_norms = []
    for x in range(12):
        if discard_tokens > 0:
            layer_norms.append(
                torch.log(
                    chosen_model.block_output["block" + str(x)]
                    .squeeze()[1:-discard_tokens, :]
                    .norm(dim=-1)
                    + 1e-6
                )
                .detach()
                .numpy()
            )
        else:
            layer_norms.append(
                torch.log(
                    chosen_model.block_output["block" + str(x)]
                    .squeeze()[1:, :]
                    .norm(dim=-1)
                    + 1e-6
                )
                .detach()
                .numpy()
            )
        # layer_norms.append(chosen_model.block_output['block'+ str(x)].squeeze()[1:,:].norm(dim = -1).detach().numpy())

    plt.figure(figsize=(12, 8))  # Adjust the width and height as needed
    for i, norms in enumerate(layer_norms):
        plt.scatter([i] * len(norms), norms, alpha=0.5)
    plt.xlabel("Layer")
    plt.ylabel("Norm Values")
    plt.show()
    return layer_norms
