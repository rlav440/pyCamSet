from numba import njit
 
@njit
def matflow(output_block, write_data):
    d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31, d32, d33, d34, d35, d36, d37, d38, d39, d40, d41, d42, d43, d44, d45, d46, d47, d48, d49, d50, d51, d52, d53, d54, d55, d56, d57, d58, d59 = output_block[0], output_block[1], output_block[2], output_block[3], output_block[4], output_block[5], output_block[6], output_block[7], output_block[8], output_block[9], output_block[10], output_block[11], output_block[12], output_block[13], output_block[14], output_block[15], output_block[16], output_block[17], output_block[18], output_block[19], output_block[20], output_block[21], output_block[22], output_block[23], output_block[24], output_block[25], output_block[26], output_block[27], output_block[28], output_block[29], output_block[30], output_block[31], output_block[32], output_block[33], output_block[34], output_block[35], output_block[36], output_block[37], output_block[38], output_block[39], output_block[40], output_block[41], output_block[42], output_block[43], output_block[44], output_block[45], output_block[46], output_block[47], output_block[48], output_block[49], output_block[50], output_block[51], output_block[52], output_block[53], output_block[54], output_block[55], output_block[56], output_block[57], output_block[58], output_block[59]
    write_data[:] = 0
    write_data[0,0] = d0
    write_data[0,1] = 1
    write_data[0,4] = d4
    write_data[0,5] = d5
    write_data[0,6] = d6
    write_data[0,7] = d7
    write_data[0,8] = d8
    write_data[0,9] = (d9*d24+d10*d33+d11*d42)
    write_data[0,10] = (d9*d25+d10*d34+d11*d43)
    write_data[0,11] = (d9*d26+d10*d35+d11*d44)
    write_data[0,12] = d9
    write_data[0,13] = d10
    write_data[0,14] = d11
    write_data[0,15] = (d9*d30+d10*d39+d11*d48)
    write_data[0,16] = (d9*d31+d10*d40+d11*d49)
    write_data[0,17] = (d9*d32+d10*d41+d11*d50)
    write_data[1,2] = d14
    write_data[1,3] = 1
    write_data[1,4] = d16
    write_data[1,5] = d17
    write_data[1,6] = d18
    write_data[1,7] = d19
    write_data[1,8] = d20
    write_data[1,9] = (d21*d24+d22*d33+d23*d42)
    write_data[1,10] = (d21*d25+d22*d34+d23*d43)
    write_data[1,11] = (d21*d26+d22*d35+d23*d44)
    write_data[1,12] = d21
    write_data[1,13] = d22
    write_data[1,14] = d23
    write_data[1,15] = (d21*d30+d22*d39+d23*d48)
    write_data[1,16] = (d21*d31+d22*d40+d23*d49)
    write_data[1,17] = (d21*d32+d22*d41+d23*d50)
