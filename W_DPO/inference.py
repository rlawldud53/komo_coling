from peft import PeftModel, PeftConfig
import transformers
import torch
from transformers import AddedToken, AutoTokenizer
import huggingface_hub
import pandas as pd
from tqdm import tqdm
import os
import argparse

def infer(input_text: str, model, tokenizer, max_length: int = 250):
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


if __name__ == '__main__':
    #create
    parser = argparse.ArgumentParser(
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tokenizer_path',type=str,default="canho/koalpaca-5.8b-3epochs-30000-data")
    parser.add_argument('--checkpoint_idx',type=int,default=1044)
    parser.add_argument('--dataset_path',type=str,default="/home/phw/work/KoMo/dataset/eval_dataset.csv")
    parser.add_argument('--save_folder',type=str,default="/home/phw/work/KoMo/inference_output/")
    
    config = parser.parse_args()
    
    
    tokenizer_path = config.tokenizer_path#"canho/koalpaca-5.8b-3epochs-30000-data"#"canho/koalpaca-5.8b-emojis-3epochs-prompt-revised"
    data_30000 = [1740]#[1044,2088,1740,1392,2436,2784,3132,3480]

    for ch_idx in tqdm(data_30000,desc="current checkpoint "):
        print(f"current checkpoint : {ch_idx}\n")
        model_name = f"jeeyoung/dpo{ch_idx}8th_trial_30000_data"
        model_kwargs = {'device_map': 'balanced'}
        policy_dtype = getattr(torch,"float32")
        peft_config = PeftConfig.from_pretrained(model_name)
        base_model_name = peft_config.base_model_name_or_path
        base_model = transformers.AutoModelForCausalLM.from_pretrained('beomi/KoAlpaca-Polyglot-5.8B', low_cpu_mem_usage=True, torch_dtype=policy_dtype, **model_kwargs)
        base_model.resize_token_embeddings(30250)
        policy = PeftModel.from_pretrained(base_model, model_name, torch_dtype=policy_dtype)
    # for ch_idx in tqdm(data_30000,desc="current checkpoint "):
    #     print(f"current checkpoint : {ch_idx}\n")
    #     model_name = f"jeeyoung/dpo{ch_idx}8th_trial_30000_data"
    #     model_kwargs = {'device_map': 'balanced'} 
    #     policy_dtype = getattr(torch,"float32")
    #     #peft_config = PeftConfig.from_pretrained("canho/koalpaca-5.8b-3epochs-30000-data")
    #     #base_model_name = peft_config.base_model_name_or_path
    #     #config.model.base_model_name = base_model_name
    #     #base_model = transformers.AutoModelForCausalLM.from_pretrained('beomi/KoAlpaca-Polyglot-5.8B', low_cpu_mem_usage=True, torch_dtype=policy_dtype, **model_kwargs)
    #     #base_model.resize_token_embeddings(30250)
    #     #policy = PeftModel.from_pretrained(base_model, model_name, torch_dtype=policy_dtype,**model_kwargs)
        
    #     policy = transformers.AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=policy_dtype)
    #     #policy.resize_token_embeddings(30250)
    #     #policy = PeftModel.from_pretrained(base_model, model_name, torch_dtype=policy_dtype)

        IGNORE_INDEX = -100
        DEFAULT_PAD_TOKEN = "[PAD]"
        DEFAULT_EOS_TOKEN = "</s>"
        DEFAULT_BOS_TOKEN = "</s>"
        DEFAULT_UNK_TOKEN = "</s>"

        emoji_tokens =  [
            "😀", "😁", "😂", "🤣", "😃", "😄", "😅", "😆", "😉", "😊", "😋", "😎", "😍", "😘", "🥰", "😗", "😙", "😚",
            "🙂", "🤗", "🤩", "🤔", "🤨", "😐", "😑", "😶", "🙄", "😏", "😣", "😥", "😮", "🤐", "😯", "😪", "😫", "🥱",
            "😴", "😌", "😛", "😜", "😝", "🤤", "😒", "😓", "😔", "😕", "🙃", "🤑", "😲", "☹", "🙁", "😖", "😞", "😟",
            "😤", "😢", "😭", "😦", "😧", "😨", "😩", "🤯", "😬", "😰", "😱", "🥵", "🥶", "😳", "🤪", "😵", "😡", "😠",
            "🤬", "😷", "🤒", "🤕", "🤢", "🤮", "🤧", "😇", "🥳", "🥺", "🤠", "🤡", "🤥", "🤫", "🤭", "🧐", "🤓", "😈",
            "👿", "👹", "👺", "💀", "👻", "👽", "👾", "🤖",


            "💌", "🕳", "💣", "💎", "🔪", "🗡", "⚔", "🛡", "🚬", "⚰", "⚱", "🏺", "🔮", "📿", "💈", "⚗", "🔭", "🔬",
            "🕯", "💡", "🔦", "🏮", "📔", "📕", "📖", "📗", "📘", "📙", "📚", "📓", "📒", "📃", "📜", "📄", "📰",
            "🗞", "📑", "🔖", "🏷", "💰", "💴", "💵", "💶", "💷", "💸", "💳", "🧾", "💹", "✉", "📧", "📨", "📩",
            "📤", "📥", "📦", "📫", "📪", "📬", "📭", "📮", "🗳", "✏", "✒", "🖋", "🖊", "🖌", "🖍", "📝", "💼",
            "📁", "📂", "🗂", "📅", "📆", "🗒", "🗓", "📇", "📈", "📉", "📊", "📋", "📌", "📍", "📎", "🖇", "📏",
            "📐", "✂", "🗃", "🗄", "🗑", "🔒", "🔓", "🔏", "🔐", "🔑", "🗝", "🔨", "🪓", "⛏", "⚒", "🛠", "🗡", "⚔",
            "🔫", "🏹", "🛡", "🔧", "🔩", "⚙", "🗜", "⚖", "🦯", "🔗", "⛓", "🧰", "🧲", "🧪", "🧫", "🧬", "🔬",
            "🔭", "📡", "💉", "💊", "🩸", "🩹", "🩺", "🚪", "🛏", "🛋", "🪑", "🚽", "🚿", "🛁", "🪒", "🧴", "🧷",
            "🧹", "🧺", "🧻", "🧼", "🪣", "🧽", "🪤", "🪒", "🔑", "🗝", "🚪", "🛌", "🛋", "🛏", "🛋", "🪑", "🚽",
            "🪣", "🛁", "🪞", "🪠", "🪤", "🪒", "🪥", "🛒", "🚬", "⚰", "⚱", "🪦", "🧿", "🪔", "🪒"
        ]
        added_emoji_tokens = [AddedToken(emoji, rstrip=False, lstrip=False, single_word=False, normalized=False, special=True) for emoji in emoji_tokens]
        peft_config = PeftConfig.from_pretrained(tokenizer_path)
        base_tokenizer_name = peft_config.base_model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(
            base_tokenizer_name,
            padding_side="right",
            model_max_length=512,
        )
        tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
        }
        )

        tokenizer.add_tokens(added_emoji_tokens)

        for token in added_emoji_tokens:
            assert tokenizer.convert_tokens_to_ids(token.content) != tokenizer.unk_token_id, f"Token {token.content} was not added correctly."
        print("All tokens added successfully.")

        policy.eval()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #policy.to(device)

        #sample_list = ['에휴 씨발 물가 좆같이 올라서 먹을 수 있는게 없노', '조선족 개새끼들 한국땅에서 못살게 존나 다 내쫓아야한다이기야', '개쌉레알이노 ㅋㅋㅋㅋ 웹툰보이가 그린웹툰들 몰입력 ㅆㅎㅌㅊ에 내용 ㅆㅎㅌㅊ 반면 웹툰작가꺼는 나 ㅆㅅㅌㅊ에 잼임', 'ㅋㅋㅋㅋ 상상속에서 맘충이랑 싸움했ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ일베에서나 키보드로 욕이나 하지 현실에선 말한마디나 제대로 하노?', '요즘 유흥에도 베트남년들 개많더라 오피에도 베트남년들 태국년들있고 대부분 국걸로 입국해서 화재들 뒷통수치고 가출할거겠지만 ㅋㅋㅋㅋㅋ']
        #sample_list += ['아 미친 존나 배고파', '교수 과제 왜 이따구로 냄?', 'ㅅㅂ 집가고 싶다']
        #sample_list += ['ㅈㄴ 게이같노 ㅋㅋ', '지랄하지말고 묻는 말에 대답이나해 이기야', '저 미친 ㄴ ㅋㅋㅋㅋㅋ', '한남 수준 실화노 ㅋㅋ']
        #sample_list += ['한남 수준 ㅉㅉ', '말하는 꼬라지만봐도 홍어 냄새 나네 라도냐?', '개슬람 다 불태워버리고 싶다', '시발 ㅈ같네']
        #sample_list = pd.read_csv("/home/phw/work/KoMo/dataset/eval_dataset.csv")['input_sentence'].to_list()
        #data_path = "/home/phw/work/KoMo/dataset/conversation_topic10.csv"
        data_path = config.dataset_path # "/home/phw/work/KoMo/dataset/eval_dataset.csv"
        sample_list = pd.read_csv(data_path)['prompt'].to_list()
        file_name = f"{config.save_folder}/{model_name.split('/')[1]}.txt"

        result = []
        save_dir = f"{config.save_folder}/{model_name.split('/')[1]}/{data_path.split('/')[-1].split('.')[0]}"
        if not os.path.exists(save_dir) : os.makedirs(save_dir)

        input_lst = []
        sample_list = ["방송에 나오는 요리사 치고 멀쩡한놈 음따이미 주방에 멀쩡한놈 자체가 없슴거의 군대지"]
        for idx, sample in tqdm(enumerate(sample_list), desc=f"current checkpoint : {ch_idx}"):
            input_lst.append(sample)
            input_text = f"""
            ### 명령어: input에 욕설, 혐오 표현이 있다면 이를 찾아 예쁜 말로 순화해주세요. output은 하나의 문장으로만 출력해주세요.
            
            ### 맥락: {sample}\n

            ### 답변:"""
            output_text = infer(input_text, policy, tokenizer)
            # print(f"Input: {input_text}")
            print(f"Output: {output_text}")

            result.append(output_text)
            #f.write(f"input : {sample}\n")
            #f.write(f"output : {output_text.strip()}\n\n")012

            if ((idx+1) % 100)==0 and idx!=0 : 
                result_df = pd.DataFrame({"input":input_lst, "output" : result})
                #result_df.to_csv(f"{save_dir}/result_0_0.csv")
                result_df.to_csv(f"{save_dir}/result_{int(idx/100)}.csv")
                #result_df.to_csv(f"{save_dir}/result_8.csv")
                input_lst, result = [],[]
    
    
    

