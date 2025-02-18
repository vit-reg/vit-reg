
def show_artifacts_paper(
    test_model: nn.Module,
    test_image: torch.Tensor,
    log_scale=False,
    token: int = 0,
    shape: tuple = (24, 24),
    discard_tokens: int = 0,
) -> None:

    test_model(test_image)
    num_blocks = len(test_model.blocks)

    print("Norm of feature values after MLP")
    output = test_model.block_output[f"block{num_blocks-1}"].squeeze(0)

    if discard_tokens > 0:
        output_norms = output.norm(dim=-1)[1:-discard_tokens]
    else:
        output_norms = output.norm(dim=-1)[1:]

    fig, ax = plt.subplots(figsize=(8, 8))  
    im = ax.imshow(output_norms.reshape(shape[0], shape[1]).detach().numpy(), cmap="viridis")
    ax.axis("off")
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    #cbar = plt.colorbar(im, cax=cax)
    #cbar.set_label("Norm Values")
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.hist(output_norms.detach().numpy(), bins=50, color="blue", alpha=0.7)
    plt.xlabel("Norm Values")
    plt.ylabel("Frequency")
    #plt.title("Histogram of Norm Values")
    plt.show()

    print("Attention maps for the last Attention Head")
    attn_map_mean = test_model.blocks[num_blocks - 1].attn.attn_map.squeeze(0).mean(dim=0)

    if discard_tokens > 0:
        attn_map_mean = attn_map_mean[token][1:-discard_tokens]
    else:
        attn_map_mean = attn_map_mean[token][1:]
    if log_scale:
        attn_map_mean = torch.log(attn_map_mean + 1e-6)

    fig, ax = plt.subplots(figsize=(8, 8))  
    im = ax.imshow(attn_map_mean.reshape(shape[0], shape[1]).detach().numpy(), cmap="viridis")
    ax.axis("off")
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    #cbar = plt.colorbar(im, cax=cax)
    #cbar.set_label("CLS Attention Map")
    plt.show()

    print("All attention maps")

    num_cols = 6
    num_rows = (num_blocks + num_cols - 1) // num_cols 
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8 * num_cols // 6, 8 * num_rows // 6)) 

    axes = axes.flatten()

    for i in range(num_blocks):
        attn_map = test_model.blocks[i].attn.attn_map.squeeze(0).mean(dim=0)
        if discard_tokens > 0:
            attn_map = attn_map[token][1:-discard_tokens]
        else:
            attn_map = attn_map[token][1:]
        attn_map_img = attn_map.reshape(shape[0], shape[1]).detach().numpy()

        im = axes[i].imshow(attn_map_img, cmap="viridis")
        axes[i].axis("off")
        axes[i].set_title(f"Block {i+1}")

        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def plot_feature_norms_with_high_attn_paper(
    chosen_model: nn.Module,
    output_tensor: torch.Tensor,
    high_attn_tokens: torch.Tensor,
    grid_size: tuple = (24, 24),
    dot_color: str = "red",
    dot_size: int = 30,
    label: str = "High Attention Tokens",
    discard_tokens: int = 0,
) -> None:

    if discard_tokens > 0:
        output = output_tensor.squeeze(0)[1:-discard_tokens]
    else:
        output = output_tensor.squeeze(0)[1:]

    output_norms = output.norm(dim=-1)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(output_norms.reshape(grid_size).detach().numpy(), cmap="viridis")
    ax.axis("off")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Norm Values")

    high_attn_indices = np.unravel_index(high_attn_tokens[0], grid_size)
    ax.scatter(
        high_attn_indices[1],
        high_attn_indices[0],
        color=dot_color,
        s=dot_size,
        label=label,
    )
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()