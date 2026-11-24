


def forward(self, L_hidden_ : torch.Tensor):
    l_hidden_ = L_hidden_
    output = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward = torch.ops.vllm.mla_forward(l_hidden_, output, 'model.layers.0.self_attn');  l_hidden_ = mla_forward = None
    output_1 = output.view(-1, 2048);  output = None
    output_2 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_1 = torch.ops.vllm.mla_forward(output_1, output_2, 'model.layers.1.self_attn');  output_1 = mla_forward_1 = None
    output_3 = output_2.view(-1, 2048);  output_2 = None
    output_4 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_2 = torch.ops.vllm.mla_forward(output_3, output_4, 'model.layers.2.self_attn');  output_3 = mla_forward_2 = None
    output_5 = output_4.view(-1, 2048);  output_4 = None
    output_6 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_3 = torch.ops.vllm.mla_forward(output_5, output_6, 'model.layers.3.self_attn');  output_5 = mla_forward_3 = None
    output_7 = output_6.view(-1, 2048);  output_6 = None
    output_8 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_4 = torch.ops.vllm.mla_forward(output_7, output_8, 'model.layers.4.self_attn');  output_7 = mla_forward_4 = None
    output_9 = output_8.view(-1, 2048);  output_8 = None
    output_10 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_5 = torch.ops.vllm.mla_forward(output_9, output_10, 'model.layers.5.self_attn');  output_9 = mla_forward_5 = None
    output_11 = output_10.view(-1, 2048);  output_10 = None
    output_12 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_6 = torch.ops.vllm.mla_forward(output_11, output_12, 'model.layers.6.self_attn');  output_11 = mla_forward_6 = None
    output_13 = output_12.view(-1, 2048);  output_12 = None
    output_14 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_7 = torch.ops.vllm.mla_forward(output_13, output_14, 'model.layers.7.self_attn');  output_13 = mla_forward_7 = None
    output_15 = output_14.view(-1, 2048);  output_14 = None
    output_16 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_8 = torch.ops.vllm.mla_forward(output_15, output_16, 'model.layers.8.self_attn');  output_15 = mla_forward_8 = None
    output_17 = output_16.view(-1, 2048);  output_16 = None
    output_18 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_9 = torch.ops.vllm.mla_forward(output_17, output_18, 'model.layers.9.self_attn');  output_17 = mla_forward_9 = None
    output_19 = output_18.view(-1, 2048);  output_18 = None
    output_20 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_10 = torch.ops.vllm.mla_forward(output_19, output_20, 'model.layers.10.self_attn');  output_19 = mla_forward_10 = None
    output_21 = output_20.view(-1, 2048);  output_20 = None
    output_22 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_11 = torch.ops.vllm.mla_forward(output_21, output_22, 'model.layers.11.self_attn');  output_21 = mla_forward_11 = None
    output_23 = output_22.view(-1, 2048);  output_22 = None
    output_24 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_12 = torch.ops.vllm.mla_forward(output_23, output_24, 'model.layers.12.self_attn');  output_23 = mla_forward_12 = None
    output_25 = output_24.view(-1, 2048);  output_24 = None
    output_26 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_13 = torch.ops.vllm.mla_forward(output_25, output_26, 'model.layers.13.self_attn');  output_25 = mla_forward_13 = None
    output_27 = output_26.view(-1, 2048);  output_26 = None
    output_28 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_14 = torch.ops.vllm.mla_forward(output_27, output_28, 'model.layers.14.self_attn');  output_27 = mla_forward_14 = None
    output_29 = output_28.view(-1, 2048);  output_28 = None
    output_30 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_15 = torch.ops.vllm.mla_forward(output_29, output_30, 'model.layers.15.self_attn');  output_29 = mla_forward_15 = None
    output_31 = output_30.view(-1, 2048);  output_30 = None
    output_32 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_16 = torch.ops.vllm.mla_forward(output_31, output_32, 'model.layers.16.self_attn');  output_31 = mla_forward_16 = None
    output_33 = output_32.view(-1, 2048);  output_32 = None
    output_34 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_17 = torch.ops.vllm.mla_forward(output_33, output_34, 'model.layers.17.self_attn');  output_33 = mla_forward_17 = None
    output_35 = output_34.view(-1, 2048);  output_34 = None
    output_36 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_18 = torch.ops.vllm.mla_forward(output_35, output_36, 'model.layers.18.self_attn');  output_35 = mla_forward_18 = None
    output_37 = output_36.view(-1, 2048);  output_36 = None
    output_38 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_19 = torch.ops.vllm.mla_forward(output_37, output_38, 'model.layers.19.self_attn');  output_37 = mla_forward_19 = None
    output_39 = output_38.view(-1, 2048);  output_38 = None
    output_40 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_20 = torch.ops.vllm.mla_forward(output_39, output_40, 'model.layers.20.self_attn');  output_39 = mla_forward_20 = None
    output_41 = output_40.view(-1, 2048);  output_40 = None
    output_42 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_21 = torch.ops.vllm.mla_forward(output_41, output_42, 'model.layers.21.self_attn');  output_41 = mla_forward_21 = None
    output_43 = output_42.view(-1, 2048);  output_42 = None
    output_44 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_22 = torch.ops.vllm.mla_forward(output_43, output_44, 'model.layers.22.self_attn');  output_43 = mla_forward_22 = None
    output_45 = output_44.view(-1, 2048);  output_44 = None
    output_46 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_23 = torch.ops.vllm.mla_forward(output_45, output_46, 'model.layers.23.self_attn');  output_45 = mla_forward_23 = None
    output_47 = output_46.view(-1, 2048);  output_46 = None
    output_48 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_24 = torch.ops.vllm.mla_forward(output_47, output_48, 'model.layers.24.self_attn');  output_47 = mla_forward_24 = None
    output_49 = output_48.view(-1, 2048);  output_48 = None
    output_50 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_25 = torch.ops.vllm.mla_forward(output_49, output_50, 'model.layers.25.self_attn');  output_49 = mla_forward_25 = None
    output_51 = output_50.view(-1, 2048);  output_50 = None
    output_52 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_26 = torch.ops.vllm.mla_forward(output_51, output_52, 'model.layers.26.self_attn');  output_51 = mla_forward_26 = None
    output_53 = output_52.view(-1, 2048);  output_52 = None
    output_54 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_27 = torch.ops.vllm.mla_forward(output_53, output_54, 'model.layers.27.self_attn');  output_53 = mla_forward_27 = None
    output_55 = output_54.view(-1, 2048);  output_54 = None
    output_56 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_28 = torch.ops.vllm.mla_forward(output_55, output_56, 'model.layers.28.self_attn');  output_55 = mla_forward_28 = None
    output_57 = output_56.view(-1, 2048);  output_56 = None
    output_58 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_29 = torch.ops.vllm.mla_forward(output_57, output_58, 'model.layers.29.self_attn');  output_57 = mla_forward_29 = None
    output_59 = output_58.view(-1, 2048);  output_58 = None
    output_60 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_30 = torch.ops.vllm.mla_forward(output_59, output_60, 'model.layers.30.self_attn');  output_59 = mla_forward_30 = None
    output_61 = output_60.view(-1, 2048);  output_60 = None
    output_62 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_31 = torch.ops.vllm.mla_forward(output_61, output_62, 'model.layers.31.self_attn');  output_61 = mla_forward_31 = None
    output_63 = output_62.view(-1, 2048);  output_62 = None
    output_64 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_32 = torch.ops.vllm.mla_forward(output_63, output_64, 'model.layers.32.self_attn');  output_63 = mla_forward_32 = None
    output_65 = output_64.view(-1, 2048);  output_64 = None
    output_66 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_33 = torch.ops.vllm.mla_forward(output_65, output_66, 'model.layers.33.self_attn');  output_65 = mla_forward_33 = None
    output_67 = output_66.view(-1, 2048);  output_66 = None
    output_68 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_34 = torch.ops.vllm.mla_forward(output_67, output_68, 'model.layers.34.self_attn');  output_67 = mla_forward_34 = None
    output_69 = output_68.view(-1, 2048);  output_68 = None
    output_70 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_35 = torch.ops.vllm.mla_forward(output_69, output_70, 'model.layers.35.self_attn');  output_69 = mla_forward_35 = None
    output_71 = output_70.view(-1, 2048);  output_70 = None
    output_72 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_36 = torch.ops.vllm.mla_forward(output_71, output_72, 'model.layers.36.self_attn');  output_71 = mla_forward_36 = None
    output_73 = output_72.view(-1, 2048);  output_72 = None
    output_74 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_37 = torch.ops.vllm.mla_forward(output_73, output_74, 'model.layers.37.self_attn');  output_73 = mla_forward_37 = None
    output_75 = output_74.view(-1, 2048);  output_74 = None
    output_76 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_38 = torch.ops.vllm.mla_forward(output_75, output_76, 'model.layers.38.self_attn');  output_75 = mla_forward_38 = None
    output_77 = output_76.view(-1, 2048);  output_76 = None
    output_78 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_39 = torch.ops.vllm.mla_forward(output_77, output_78, 'model.layers.39.self_attn');  output_77 = mla_forward_39 = None
    output_79 = output_78.view(-1, 2048);  output_78 = None
    output_80 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_40 = torch.ops.vllm.mla_forward(output_79, output_80, 'model.layers.40.self_attn');  output_79 = mla_forward_40 = None
    output_81 = output_80.view(-1, 2048);  output_80 = None
    output_82 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_41 = torch.ops.vllm.mla_forward(output_81, output_82, 'model.layers.41.self_attn');  output_81 = mla_forward_41 = None
    output_83 = output_82.view(-1, 2048);  output_82 = None
    output_84 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_42 = torch.ops.vllm.mla_forward(output_83, output_84, 'model.layers.42.self_attn');  output_83 = mla_forward_42 = None
    output_85 = output_84.view(-1, 2048);  output_84 = None
    output_86 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_43 = torch.ops.vllm.mla_forward(output_85, output_86, 'model.layers.43.self_attn');  output_85 = mla_forward_43 = None
    output_87 = output_86.view(-1, 2048);  output_86 = None
    output_88 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_44 = torch.ops.vllm.mla_forward(output_87, output_88, 'model.layers.44.self_attn');  output_87 = mla_forward_44 = None
    output_89 = output_88.view(-1, 2048);  output_88 = None
    output_90 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_45 = torch.ops.vllm.mla_forward(output_89, output_90, 'model.layers.45.self_attn');  output_89 = mla_forward_45 = None
    output_91 = output_90.view(-1, 2048);  output_90 = None
    output_92 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_46 = torch.ops.vllm.mla_forward(output_91, output_92, 'model.layers.46.self_attn');  output_91 = mla_forward_46 = None
    output_93 = output_92.view(-1, 2048);  output_92 = None
    output_94 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_47 = torch.ops.vllm.mla_forward(output_93, output_94, 'model.layers.47.self_attn');  output_93 = mla_forward_47 = None
    output_95 = output_94.view(-1, 2048);  output_94 = None
    output_96 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_48 = torch.ops.vllm.mla_forward(output_95, output_96, 'model.layers.48.self_attn');  output_95 = mla_forward_48 = None
    output_97 = output_96.view(-1, 2048);  output_96 = None
    output_98 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_49 = torch.ops.vllm.mla_forward(output_97, output_98, 'model.layers.49.self_attn');  output_97 = mla_forward_49 = None
    output_99 = output_98.view(-1, 2048);  output_98 = None
    output_100 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_50 = torch.ops.vllm.mla_forward(output_99, output_100, 'model.layers.50.self_attn');  output_99 = mla_forward_50 = None
    output_101 = output_100.view(-1, 2048);  output_100 = None
    output_102 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_51 = torch.ops.vllm.mla_forward(output_101, output_102, 'model.layers.51.self_attn');  output_101 = mla_forward_51 = None
    output_103 = output_102.view(-1, 2048);  output_102 = None
    output_104 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_52 = torch.ops.vllm.mla_forward(output_103, output_104, 'model.layers.52.self_attn');  output_103 = mla_forward_52 = None
    output_105 = output_104.view(-1, 2048);  output_104 = None
    output_106 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_53 = torch.ops.vllm.mla_forward(output_105, output_106, 'model.layers.53.self_attn');  output_105 = mla_forward_53 = None
    output_107 = output_106.view(-1, 2048);  output_106 = None
    output_108 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_54 = torch.ops.vllm.mla_forward(output_107, output_108, 'model.layers.54.self_attn');  output_107 = mla_forward_54 = None
    output_109 = output_108.view(-1, 2048);  output_108 = None
    output_110 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_55 = torch.ops.vllm.mla_forward(output_109, output_110, 'model.layers.55.self_attn');  output_109 = mla_forward_55 = None
    output_111 = output_110.view(-1, 2048);  output_110 = None
    output_112 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_56 = torch.ops.vllm.mla_forward(output_111, output_112, 'model.layers.56.self_attn');  output_111 = mla_forward_56 = None
    output_113 = output_112.view(-1, 2048);  output_112 = None
    output_114 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_57 = torch.ops.vllm.mla_forward(output_113, output_114, 'model.layers.57.self_attn');  output_113 = mla_forward_57 = None
    output_115 = output_114.view(-1, 2048);  output_114 = None
    output_116 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_58 = torch.ops.vllm.mla_forward(output_115, output_116, 'model.layers.58.self_attn');  output_115 = mla_forward_58 = None
    output_117 = output_116.view(-1, 2048);  output_116 = None
    output_118 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_59 = torch.ops.vllm.mla_forward(output_117, output_118, 'model.layers.59.self_attn');  output_117 = mla_forward_59 = None
    output_119 = output_118.view(-1, 2048);  output_118 = None
    output_120 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_60 = torch.ops.vllm.mla_forward(output_119, output_120, 'model.layers.60.self_attn');  output_119 = mla_forward_60 = None
    output_121 = output_120.view(-1, 2048);  output_120 = None
    output_122 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_61 = torch.ops.vllm.mla_forward(output_121, output_122, 'model.layers.61.self_attn');  output_121 = mla_forward_61 = None
    output_123 = output_122.view(-1, 2048);  output_122 = None
    output_124 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_62 = torch.ops.vllm.mla_forward(output_123, output_124, 'model.layers.62.self_attn');  output_123 = mla_forward_62 = None
    output_125 = output_124.view(-1, 2048);  output_124 = None
    output_126 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_63 = torch.ops.vllm.mla_forward(output_125, output_126, 'model.layers.63.self_attn');  output_125 = mla_forward_63 = None
    output_127 = output_126.view(-1, 2048);  output_126 = None
    output_128 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_64 = torch.ops.vllm.mla_forward(output_127, output_128, 'model.layers.64.self_attn');  output_127 = mla_forward_64 = None
    output_129 = output_128.view(-1, 2048);  output_128 = None
    output_130 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_65 = torch.ops.vllm.mla_forward(output_129, output_130, 'model.layers.65.self_attn');  output_129 = mla_forward_65 = None
    output_131 = output_130.view(-1, 2048);  output_130 = None
    output_132 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_66 = torch.ops.vllm.mla_forward(output_131, output_132, 'model.layers.66.self_attn');  output_131 = mla_forward_66 = None
    output_133 = output_132.view(-1, 2048);  output_132 = None
    output_134 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_67 = torch.ops.vllm.mla_forward(output_133, output_134, 'model.layers.67.self_attn');  output_133 = mla_forward_67 = None
    output_135 = output_134.view(-1, 2048);  output_134 = None
    output_136 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_68 = torch.ops.vllm.mla_forward(output_135, output_136, 'model.layers.68.self_attn');  output_135 = mla_forward_68 = None
    output_137 = output_136.view(-1, 2048);  output_136 = None
    output_138 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_69 = torch.ops.vllm.mla_forward(output_137, output_138, 'model.layers.69.self_attn');  output_137 = mla_forward_69 = None
    output_139 = output_138.view(-1, 2048);  output_138 = None
    output_140 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_70 = torch.ops.vllm.mla_forward(output_139, output_140, 'model.layers.70.self_attn');  output_139 = mla_forward_70 = None
    output_141 = output_140.view(-1, 2048);  output_140 = None
    output_142 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_71 = torch.ops.vllm.mla_forward(output_141, output_142, 'model.layers.71.self_attn');  output_141 = mla_forward_71 = None
    output_143 = output_142.view(-1, 2048);  output_142 = None
    output_144 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_72 = torch.ops.vllm.mla_forward(output_143, output_144, 'model.layers.72.self_attn');  output_143 = mla_forward_72 = None
    output_145 = output_144.view(-1, 2048);  output_144 = None
    output_146 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_73 = torch.ops.vllm.mla_forward(output_145, output_146, 'model.layers.73.self_attn');  output_145 = mla_forward_73 = None
    output_147 = output_146.view(-1, 2048);  output_146 = None
    output_148 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_74 = torch.ops.vllm.mla_forward(output_147, output_148, 'model.layers.74.self_attn');  output_147 = mla_forward_74 = None
    output_149 = output_148.view(-1, 2048);  output_148 = None
    output_150 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_75 = torch.ops.vllm.mla_forward(output_149, output_150, 'model.layers.75.self_attn');  output_149 = mla_forward_75 = None
    output_151 = output_150.view(-1, 2048);  output_150 = None
    output_152 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_76 = torch.ops.vllm.mla_forward(output_151, output_152, 'model.layers.76.self_attn');  output_151 = mla_forward_76 = None
    output_153 = output_152.view(-1, 2048);  output_152 = None
    output_154 = torch.empty((8, 2048), dtype = torch.bfloat16, device = device(type='cpu'))
    mla_forward_77 = torch.ops.vllm.mla_forward(output_153, output_154, 'model.layers.77.self_attn');  output_153 = mla_forward_77 = None
    output_155 = output_154.view(-1, 2048);  output_154 = None
    return (output_155,)
    