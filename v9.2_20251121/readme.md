* my_train.py：训练文件，运行./scripts/my_train.sh。训练的模型权重保存在outputs_my_train目录
* my_sample_all.py：推理文件，运行./scripts/my_sample_all.sh，生成测试集所有字体的某一个字用于肉眼对比，测试结果图保存在outputs_my_test目录
* src/：各模块代码目录
  * diffusion/：diffusion模型
  * dpm_solver/：diffusion推理代码框架
  * embedding/：图像、部首、骨架、贝塞尔等模态的编码器解码器相关模块
  * main/：my_train, my_sample训练推理文件直接调用的一些模块
  * modules/：controlnet，transformer，unet，vae等基础模块
  * multi-modal/：多模态对齐融合模块