def run_evaluation(model, valid_dl, device):
    model.eval()
    all_losses = []
    
    progress_bar = tqdm(valid_dl, desc="Evaluating", leave=False)

    for batch in progress_bar:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            output = model(**batch)
            all_losses.append(output.loss.item())

    return {"valid_loss" : np.mean(all_losses)}
    