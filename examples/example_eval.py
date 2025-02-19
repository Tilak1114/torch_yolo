# from torchyolo.yolo.data.dataset import YOLODataModule
# from torchyolo.yolo.yolo_v11_module import YOLOv11Module

# if __name__ == "__main__":
#     data_module = YOLODataModule(
#         dataset_yaml="/data/tilak/projects/gtf/dataset_yolo/dataset.yaml",
#         batch_size=64,
#     )
#     val_dataloader = data_module.val_dataloader()

#     finetuned_module = YOLOv11Module(
#         ckpt_path="/data/tilak/projects/gtf/output/ft_fshn/weights/final.pt",
#         save_dir="/data/tilak/projects/gtf/eval/ft_fshn",
#     )

#     finetuned_evaluation_results = finetuned_module.evaluate_dataloader(
#         dataloader=val_dataloader, conf_thres=0.5
#     )

#     print(finetuned_evaluation_results)
