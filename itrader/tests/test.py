import llama_cpp as ll
llm = ll.Llama(
    model_path="C:\Users\Admin\OneDrive - Budapesti Műszaki és Gazdaságtudományi Egyetem\Documents\iTRADER\itrader-azr\itrader\models\Deepseek-6.7B-Q4_K_M\deepseek_6b_q4_k_m.gguf",
    n_gpu_layers=-1,         # put everything on GPU
    n_ctx=4096,
)
print(llm("Q: 2+2?\nA: ").strip())

