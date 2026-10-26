# Current-study verification

Run `python -m unittest discover -s tests -v` after completing all three notebooks. The tests inspect current artifacts and reload predictions; they do not train new models.

Checks cover source/input hashes, a passed fresh reconstruction audit, disjoint partitions, source overlap, stage order, fresh initialization, fine-tuning ancestry, distinct independent models, input sensitivity, every epoch, patience/optimizer state, checkpoint hashes, validation-only selection, reloaded probabilities, recomputed metrics, and executed notebook structure.

Float32 probability CSVs are parsed back to their original dtype before comparison; treating short float32 decimal representations as exact float64 values would create meaningless serialization discrepancies.
