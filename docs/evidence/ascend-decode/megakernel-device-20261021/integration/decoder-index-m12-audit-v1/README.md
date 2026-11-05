# Full-decoder C8 index-key mismatch

The original asynchronous two-layer fixture passes M1/M2 for eight changing inputs each, then fails C8 M12 iteration 0. The [device report](../vendor-hook-decoder212-storage-forward-int8-wide-v2-a3-194/decoder-forward.json) compares the original DecoderLayer 6/7 with native attention installed; both arms retain the original FFN and explicit TP cut inputs.

The [saved-output audit](audit.json) locates one INT8 cache difference at `[5,121,0,31]`: original 9, native 8. This is physical slot 761, exactly the current input token 5 at logical position 8185. The entire index scale cache matches. Top-k row 5 swaps positions 1915/1916; row 6 rotates positions 1075–1077. Both changes move key 8185, both rows retain the same complete 2048-element multiset, and layer 7 retains layer 6's ordered result. The other 22 compared boundaries match raw bytes.

These observations establish a current-token key difference and ordered selection differences. They do not establish equal scores or identify the first arithmetic difference. The saved fixture lacks index query codes, query scales, and head weights. The next diagnostic retains the actual WK, layer norm, RoPE, Hadamard, and dynamic-quantization boundaries inside the unchanged full decoder. It must reproduce the complete parent inputs and outputs before its retained values can explain this failure.

The fixture SHA256 is `a0e416ae17b20ea8fcf4df2b4c732dbf143da9ed520549a2cb09d71bfec1673b`. Its local path and the reproducible audit script SHA are recorded in `audit.json`; the tensor file remains outside the repository.
