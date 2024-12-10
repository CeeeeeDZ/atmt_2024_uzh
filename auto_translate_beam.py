import subprocess
import time
import re

with open("assignments/05/bleu_score.txt", "w") as out_file:

    for beam_size in range(1, 26):
        #make directory for each beam size
        subprocess.run(["mkdir", f"assignments/05/beam_{beam_size}"])

        start = time.time()
        #run beam search program
        subprocess.run(["python", "translate_beam.py",
                        "--data", "data/en-fr/prepared",
                        "--dicts", "data/en-fr/prepared",
                        "--checkpoint-path", "assignments/03/with_batch_80/checkpoints/checkpoint_last.pt",
                        "--output", f"assignments/05/beam_{beam_size}/model_translations.txt",
                        "--beam-size", str(beam_size)])
        end = time.time()
    
        #run postprocess script
        subprocess.run(["bash", "scripts/postprocess.sh",
                        f"assignments/05/beam_{beam_size}/model_translations.txt",
                        f"assignments/05/beam_{beam_size}/model_translations.p.txt",
                        "en"])
    
        #run sacrebleu
        translated_txt = subprocess.Popen(['cat', f"assignments/05/beam_{beam_size}/model_translations.p.txt"], stdout=subprocess.PIPE)
        sacrebleu_process = subprocess.run(["sacrebleu", "data/en-fr/raw/test.en"], stdin=translated_txt.stdout, stdout=subprocess.PIPE, text=True)
        translated_txt.stdout.close()

        #get execution time
        execution_time = f"Execution time: {end - start} seconds"

        #get bleu score and BP
        score = re.search(r"score\": ([0-9\.]+),", sacrebleu_process.stdout)
        BP = re.search(r"BP = ([0-9\.]+) ", sacrebleu_process.stdout)

        with open(f"assignments/05/beam_{beam_size}/eval_results.txt", "w") as f:
            f.write(f"Beam size {beam_size}\n")
            f.write(sacrebleu_process.stdout)
            f.write(execution_time)
            f.write(f"\nBLEU score: {score.group(1)}\n")
            f.write(f"BP: {BP.group(1)}\n")

        out_file.write(f"{beam_size}, {score.group(1)}, {BP.group(1)}, {(end - start):.2f}\n")
    
        print(f"Beam size {beam_size}")
        print(sacrebleu_process.stdout)
        print(execution_time)
        print(score.group(1))
        print(BP.group(1))

