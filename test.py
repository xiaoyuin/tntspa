
vocab_en = open("data/monument_600/vocab.en")
vocab_sparql = open("data/monument_600/vocab.sparql")
vocab_shared = open("data/monument_600/vocab.shared", "w")

vocabs = set()
for line in vocab_en:
    vocabs.add(line)
for line in vocab_sparql:
    vocabs.add(line)
vocab_shared.writelines(vocabs)
vocab_shared.flush()

vocab_en.close()
vocab_sparql.close()
vocab_shared.close()