import os
import torch

def inspect_model_file(model, optimizer ,  prefix , filepath):
    """
    打印模型参数的关键信息：dtype, device, norm, std, min, max
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        sep = "=" * 100
        f.write(f"\n{sep}\n")
        f.write(f"[{prefix}] 模型参数诊断\n")
        f.write(f"{sep}\n")
        
        header = f"{'参数名':<55s} {'dtype':<15s} {'norm':>10s} {'std':>10s} {'min':>10s} {'max':>10s}"
        f.write(header)
        f.write("\n")
        f.write("-" * (len(header) + 10))
        f.write("\n")
        
        for name, param in model.named_parameters():
            p = param.data
            dtype_str = str(p.dtype).replace("torch.", "")
            line = (
                f"{name:<55s} "
                f"{dtype_str:<15s} "
                f"{p.norm().item():10.4f} "
                f"{p.std().item():10.4f} "
                f"{p.min().item():10.4f} "
                f"{p.max().item():10.4f}"
            )
            f.write(line)
            f.write("\n")

        optimizer_state_count = getattr(optimizer, 'state', None)
        if optimizer_state_count is not None:
            f.write(f"optimizer state 条目数: {len(optimizer.state)}")    
            f.write("\n")

        # 汇总
        total_params = sum(p.numel() for p in model.parameters())
        f.write(f"\n总参数量: {total_params:,}")
        f.write("\n")
        f.write(f"{sep}\n")
        f.write("\n")


def inspect_model(model, optimizer , prefix):

    """
    打印模型参数的关键信息：dtype, device, norm, std, min, max
    """
    sep = "=" * 100
    print(f"\n{sep}")
    print(f"[{prefix}] 模型参数诊断")
    print(f"{sep}")
    
    header = f"{'参数名':<55s} {'dtype':<15s} {'norm':>10s} {'std':>10s} {'min':>10s} {'max':>10s}"
    print(header)
    print("-" * (len(header) + 10))
    
    for name, param in model.named_parameters():
        p = param.data
        dtype_str = str(p.dtype).replace("torch.", "")
        line = (
            f"{name:<55s} "
            f"{dtype_str:<15s} "
            f"{p.norm().item():10.4f} "
            f"{p.std().item():10.4f} "
            f"{p.min().item():10.4f} "
            f"{p.max().item():10.4f}"
        )
        print(line)

    optimizer_state_count = getattr(optimizer, 'state', None)
    if optimizer_state_count is not None:
        print(f"optimizer state 条目数: {len(optimizer.state)}")    
    # 汇总
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params:,}")
    print(f"{sep}\n")


def print_and_write_para(model , optimizer , prefix  ,filepath ):
    inspect_model(model , optimizer , prefix)
    inspect_model_file (model , optimizer , prefix , filepath)


def save_and_print_and_write_para(model , optimizer , prefix  ,filepath  , num_of_copies , global_step , best_validation_loss , lr_scheduler , checkpoint_dir , file_name , file_suffix , prompt):
    for i in range(num_of_copies):
        torch.save({
            'step': global_step,
            'val_loss': best_validation_loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict' : lr_scheduler.state_dict(),
            }, os.path.join(
                checkpoint_dir , (
                    file_name + str(i) + file_suffix
                    )
                ))
    print("Model is saved!")
    print(prompt)
    inspect_model(model , optimizer , prefix)
    inspect_model_file (model , optimizer , prefix , filepath)

