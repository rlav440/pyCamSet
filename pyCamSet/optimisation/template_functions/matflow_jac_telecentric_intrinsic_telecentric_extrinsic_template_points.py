from numba import njit
 
@njit
def matflow(output_block, write_data):
    d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31, d32, d33, d34, d35, d36, d37, d38, d39, d40, d41, d42, d43, d44, d45, d46, d47, d48, d49, d50, d51, d52, d53 = output_block[0], output_block[1], output_block[2], output_block[3], output_block[4], output_block[5], output_block[6], output_block[7], output_block[8], output_block[9], output_block[10], output_block[11], output_block[12], output_block[13], output_block[14], output_block[15], output_block[16], output_block[17], output_block[18], output_block[19], output_block[20], output_block[21], output_block[22], output_block[23], output_block[24], output_block[25], output_block[26], output_block[27], output_block[28], output_block[29], output_block[30], output_block[31], output_block[32], output_block[33], output_block[34], output_block[35], output_block[36], output_block[37], output_block[38], output_block[39], output_block[40], output_block[41], output_block[42], output_block[43], output_block[44], output_block[45], output_block[46], output_block[47], output_block[48], output_block[49], output_block[50], output_block[51], output_block[52], output_block[53]
    write_data[:] = 0
    write_data[0,0] = d0
    write_data[0,1] = 1
    write_data[0,2] = d2
    write_data[0,4] = d4
    write_data[0,5] = d5
    write_data[0,6] = (d6*d18+d7*d24+d8*d30)
    write_data[0,7] = (d6*d19+d7*d25+d8*d31)
    write_data[0,8] = (d6*d20+d7*d26+d8*d32)
    write_data[0,9] = ((d6*d21+d7*d27+d8*d33)*d36+(d6*d22+d7*d28+d8*d34)*d42+(d6*d23+d7*d29+d8*d35)*d48)
    write_data[0,10] = ((d6*d21+d7*d27+d8*d33)*d37+(d6*d22+d7*d28+d8*d34)*d43+(d6*d23+d7*d29+d8*d35)*d49)
    write_data[0,11] = ((d6*d21+d7*d27+d8*d33)*d38+(d6*d22+d7*d28+d8*d34)*d44+(d6*d23+d7*d29+d8*d35)*d50)
    write_data[0,12] = (d6*d21+d7*d27+d8*d33)
    write_data[0,13] = (d6*d22+d7*d28+d8*d34)
    write_data[0,14] = (d6*d23+d7*d29+d8*d35)
    write_data[1,0] = d9
    write_data[1,2] = d11
    write_data[1,3] = 1
    write_data[1,4] = d13
    write_data[1,5] = d14
    write_data[1,6] = (d15*d18+d16*d24+d17*d30)
    write_data[1,7] = (d15*d19+d16*d25+d17*d31)
    write_data[1,8] = (d15*d20+d16*d26+d17*d32)
    write_data[1,9] = ((d15*d21+d16*d27+d17*d33)*d36+(d15*d22+d16*d28+d17*d34)*d42+(d15*d23+d16*d29+d17*d35)*d48)
    write_data[1,10] = ((d15*d21+d16*d27+d17*d33)*d37+(d15*d22+d16*d28+d17*d34)*d43+(d15*d23+d16*d29+d17*d35)*d49)
    write_data[1,11] = ((d15*d21+d16*d27+d17*d33)*d38+(d15*d22+d16*d28+d17*d34)*d44+(d15*d23+d16*d29+d17*d35)*d50)
    write_data[1,12] = (d15*d21+d16*d27+d17*d33)
    write_data[1,13] = (d15*d22+d16*d28+d17*d34)
    write_data[1,14] = (d15*d23+d16*d29+d17*d35)
