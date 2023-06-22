#training_script: 
for l  in range(0,args.add_fc +2):
    size_loss = min(feat_source[l].size(0), feat_target[l].size(0)) 
    total_fc_L2norm_loss = 0
    feat_source_sel = feat_source[l][:size_loss]
    feat_target_sel = feat_target[l][:size_loss]

    # break into multiple batches to avoid "out of memory" issue
    size_batch = min(256, feat_source_sel.size(0))
    feat_source_sel = feat_source_sel.view((-1, size_batch) + feat_source_sel.size()[1:])
    feat_target_sel = feat_target_sel.view((-1, size_batch) + feat_target_sel.size()[1:])
    size_loss = min(feat_source_sel[l].size(0), feat_target_sel[l].size(0))  # choose the smaller number
    for t in range(size_loss):
        s_fc2_L2norm_loss = get_L2norm_loss_self_driven(feat_source_sel[t]) #May be remove the shared layers ? feat_source[:-args.add_fc]
        t_fc2_L2norm_loss = get_L2norm_loss_self_driven(feat_target_sel[t])
        local_fc_L2norm_loss =  s_fc2_L2norm_loss + t_fc2_L2norm_loss
        total_fc_L2norm_loss += local_fc_L2norm_loss

    total_fc_L2norm_loss = total_fc_L2norm_loss/size_loss
    loss_classification += total_fc_L2norm_loss


#L2_ loss func
def get_L2norm_loss_self_driven(x):
    weight_L2norm = 0.05
    radius = x.norm(p=2, dim=1).detach()
    assert radius.requires_grad == False
    radius = radius + 1.0
    l = ((x.norm(p=2, dim=1) - radius) ** 2).mean()
    return weight_L2norm * l

def MCC_entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy