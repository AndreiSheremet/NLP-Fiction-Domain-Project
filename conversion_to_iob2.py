import csv
input_file="Fiction Dataset - All data.csv"
output_file="fiction_dataset.iob2"

with open(input_file, newline='', encoding="cp1252") as csvfile:
    reader= csv.reader(csvfile, delimiter=';')

    current_sent_id=None

    with open(output_file,"w", encoding="cp1252") as out:
        token_index = 1
        for row in reader:
            if len(row) < 4:
                continue
            sent_id= row[3].strip()
            token= row[1].strip()
            label= row[2].strip()

            if not sent_id or not token or not label:
                continue

            if current_sent_id is not None and sent_id != current_sent_id:
                token_index =1
                out.write("\n")

            current_sent_id = sent_id

            if token == "APOS":
                token="'"

            out.write(f"{token_index}\t{token}\t{label}\n")
            token_index +=1
print("Done! Saved to fiction_dataset.iob2")
        



