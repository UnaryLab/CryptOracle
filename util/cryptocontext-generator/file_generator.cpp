//==================================================================================
// BSD 2-Clause License
//
// Copyright (c) 2014-2022, NJIT, Duality Technologies Inc. and other contributors
//
// All rights reserved.
//
// Author TPOC: contact@openfhe.org
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//==================================================================================


#include <iomanip>
#include <tuple>
#include <unistd.h>
#include <cmath>
#include <random>
#include <chrono>
#include <map>

#include "openfhe.h"

// header files needed for serialization
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"

using namespace lbcrypto;

/////////////////////////////////////////////////////////////////
// NOTE:
// If running locally, you may want to replace the "hardcoded" DATAFOLDER with
// the DATAFOLDER location below which gets the current working directory
/////////////////////////////////////////////////////////////////
// char buff[1024];
// std::string DATAFOLDER = std::string(getcwd(buff, 1024));

// Save-Load locations for keys
const std::string DATAFOLDER = "serialized-files";
std::string ccLocation       = "/cryptocontext.txt";
std::string pubKeyLocation   = "/key_pub.txt";   // Pub key
std::string multKeyLocation  = "/key_mult.txt";  // relinearization key
std::string rotKeyLocation   = "/key_rot.txt";   // automorphism / rotation key
std::string bootstrapKeyLocation   = "/key_bootstrap.txt";   

// Save-load locations for RAW ciphertexts
std::string cipherOneLocation = "/a_input_ciphertext.txt";
std::string cipherTwoLocation = "/b_input_ciphertext.txt";
std::string cipherThreeLocation = "/c_input_ciphertext.txt";
std::string cipherOutLocation = "/output_ciphertext.txt";

// Save-load locations for evaluated ciphertexts
std::string cipherMultLocation   = "/ciphertextMult.txt";
std::string cipherAddLocation    = "/ciphertextAdd.txt";
std::string cipherRotLocation    = "/ciphertextRot.txt";
std::string cipherRotNegLocation = "/ciphertextRotNegLocation.txt";
std::string clientVectorLocation = "/ciphertextVectorFromClient.txt";

std::vector<double> generateRandomVector(int batchSize) {
    std::vector<double> vec(batchSize);
    
    unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> distribution(0.0, 100);

    for (int i = 0; i < batchSize; ++i) {
        vec[i] = distribution(generator);
    }
    return vec;
}

std::string to_lower(const std::string& str) {
    std::string lower_str = str;
    std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return lower_str;
}

/*std::tuple<CryptoContext<DCRTPoly>, KeyPair<DCRTPoly>, int> */ 
void fileSetupAndWrite(const std::string& securityLevel, uint32_t N, uint32_t batchSize, uint32_t depth, bool generateBootstrapKey) {
    // Set main params
    const int scaleModSize = 48; //25;// 48; //50
    const int firstModeSize = 52; //30 // 52; //51

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetFirstModSize(firstModeSize);
    parameters.SetScalingTechnique(lbcrypto::ScalingTechnique::FLEXIBLEAUTO);

    // Set security level based on input parameter
    if (securityLevel != "none") {
        std::map<std::string, SecurityLevel> securityMap = {
            {"128c", HEStd_128_classic},
            {"192c", HEStd_192_classic},
            {"256c", HEStd_256_classic},
            {"128q", HEStd_128_quantum},
            {"192q", HEStd_192_quantum},
            {"256q", HEStd_256_quantum}
        };
        auto it = securityMap.find(securityLevel);
        if (it != securityMap.end()) {
            parameters.SetSecurityLevel(it->second);
        } else {
            std::cout << "Invalid security standard. Using default value of HEStd_NotSet." << std::endl;
            parameters.SetSecurityLevel(HEStd_NotSet);
        }
    } else {
        parameters.SetSecurityLevel(HEStd_NotSet);
    }

    if (parameters.GetSecurityLevel() == HEStd_NotSet) {
        parameters.SetRingDim(N);
    }

    parameters.SetBatchSize(batchSize);

    // usint dnum = 3;
    // std::vector<uint32_t> levelBudget = {3,3};
    std::vector<uint32_t> levelBudget = {4,4};
    // Adjust level budget based on security level and ring dimension (N)
    // if (N >= 16 || securityLevel != "none") {
        //     levelBudget = {4,4}; // Adjust the level budget to avoid exceeding the correction factor
        // }
        std::vector<uint32_t> bsgsDim = {0,0};  // Try non-zero BSGS dimensions for better compatibility
        
    parameters.SetNumLargeDigits(2);
    // parameters.SetKeySwitchTechnique(KeySwitchTechnique::HYBRID);
    
    if (generateBootstrapKey){
        uint32_t approxBootstrapDepth = 4 + 4; // During EvalRaise, Chebyshev, DoubleAngle
        uint32_t levelsUsedBeforeBootstrap = depth;
        usint bootstrap_depth = levelsUsedBeforeBootstrap + FHECKKSRNS::GetBootstrapDepth(approxBootstrapDepth, levelBudget, SecretKeyDist::SPARSE_TERNARY);
        parameters.SetMultiplicativeDepth(bootstrap_depth);
    } 
    else {
        parameters.SetMultiplicativeDepth(depth);
    }

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    if (generateBootstrapKey) {
        cc->EvalBootstrapSetup(levelBudget, bsgsDim, batchSize, 0, true);
    }

    std::cout << "Cryptocontext generated" << std::endl;

    KeyPair<DCRTPoly> keyPair = cc->KeyGen();
    std::cout << "Keypair generated" << std::endl;

    cc->EvalMultKeyGen(keyPair.secretKey);
    std::cout << "Eval Mult Keys/ Relinearization keys have been generated" << std::endl;

    // Rotation keys necessary for matrix multiplications
    cc->EvalRotateKeyGen(keyPair.secretKey, { 1, 2 ,4 ,6 ,8 ,10 ,12 ,14 ,16 ,18 ,20 ,22 ,24 ,26 ,28 ,30 ,32 ,34 ,36 ,38 ,40 ,42 ,44 ,46 ,48 ,50 ,52 ,54 ,56 ,58 ,60 ,62,64, 96, 128 ,160,
    192, 224, 256 ,320, 384 ,448, 512 ,576, 640 ,704, 768 ,832, 896 ,960, 1024 ,1152 ,1280 ,1408 ,1536 ,1664 ,1792 ,1920 ,2048 ,2176 ,2304 ,2432 ,2560 ,2688 ,2816 ,2944 ,3072 ,3200 ,
    3328 ,3456 ,3584 ,3712 ,3840 ,3968,4096,-1 ,-2 ,-4 ,-8 ,-12, -15, -16 ,-32 ,-56, -63, -64 ,-128 ,-240, -255, -256 ,-512 ,-992, -1023, -1024 ,-2048,-4032,-4095});
    // cc->EvalRotateKeyGen(keyPair.secretKey, {1024});
    std::cout << "Rotation keys generated" << std::endl;

    if (generateBootstrapKey) {
        cc->EvalBootstrapKeyGen(keyPair.secretKey, batchSize);
        std::cout << "Bootstrapping key generated" << std::endl;
    }

    std::vector<double> vec1 = generateRandomVector(batchSize);
    Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(vec1, 1, 0, nullptr, batchSize);
    ptxt1->SetLength(batchSize);
    
    std::vector<double> vec2 = generateRandomVector(batchSize);
    Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(vec2, 1, 0, nullptr, batchSize);
    ptxt2->SetLength(batchSize);

    std::vector<double> vec3 = generateRandomVector(batchSize);
    Plaintext ptxt3 = cc->MakeCKKSPackedPlaintext(vec3, 1, 0, nullptr, batchSize);
    ptxt3->SetLength(batchSize);
    // std::cout << "Plaintext version of first vector: " << ptxt1 << std::endl;
    
    std::cout << "Plaintexts have been generated from vectors" << std::endl;
    
    auto serverC1 = cc->Encrypt(keyPair.publicKey, ptxt1);
    auto serverC2 = cc->Encrypt(keyPair.publicKey, ptxt2);
    auto serverC3 = cc->Encrypt(keyPair.publicKey, ptxt3);
    std::cout << "Ciphertexts have been generated from Plaintexts" << std::endl;

    if (!Serial::SerializeToFile(DATAFOLDER + ccLocation, cc, SerType::BINARY)) {
        std::cerr << "Error writing serialization of the crypto context to "
                     "cryptocontext.txt"
                  << std::endl;
        std::exit(1);
    }

    std::cout << "Cryptocontext serialized" << std::endl;

    if (!Serial::SerializeToFile(DATAFOLDER + pubKeyLocation, keyPair.publicKey, SerType::BINARY)) {
        std::cerr << "Exception writing public key to pubkey.txt" << std::endl;
        std::exit(1);
    }
    std::cout << "Public key serialized" << std::endl;

    std::ofstream multKeyFile(DATAFOLDER + multKeyLocation, std::ios::out | std::ios::binary);
    if (multKeyFile.is_open()) {
        if (!cc->SerializeEvalMultKey(multKeyFile, SerType::BINARY)) {
            std::cerr << "Error writing eval mult keys" << std::endl;
            std::exit(1);
        }
        std::cout << "EvalMult/relinearization keys have been serialized" << std::endl;
        multKeyFile.close();
    }
    else {
        std::cerr << "Error serializing EvalMult keys" << std::endl;
        std::exit(1);
    }

    std::ofstream rotationKeyFile(DATAFOLDER + rotKeyLocation, std::ios::out | std::ios::binary);
    if (rotationKeyFile.is_open()) {
        if (!cc->SerializeEvalAutomorphismKey(rotationKeyFile, SerType::BINARY)) {
            std::cerr << "Error writing rotation keys" << std::endl;
            std::exit(1);
        }
        std::cout << "Rotation keys have been serialized" << std::endl;
    }
    else {
        std::cerr << "Error serializing Rotation keys" << std::endl;
        std::exit(1);
    }

    if (generateBootstrapKey){
        std::ofstream bootstrapKeyFile(DATAFOLDER + bootstrapKeyLocation, std::ios::out | std::ios::binary);
        if (bootstrapKeyFile.is_open()) {
            if (!cc->SerializeEvalAutomorphismKey(bootstrapKeyFile, SerType::BINARY)) {
                std::cerr << "Error writing bootstrapping keys" << std::endl;
                std::exit(1);
            }
            std::cout << "Bootstrapping keys have been serialized" << std::endl;
        }
        else {
            std::cerr << "Error serializing Bootstrapping keys" << std::endl;
            std::exit(1);
        }
    }   

    if (!Serial::SerializeToFile(DATAFOLDER + cipherOneLocation, serverC1, SerType::BINARY)) {
        std::cerr << " Error writing ciphertext 1" << std::endl;
    }

    if (!Serial::SerializeToFile(DATAFOLDER + cipherTwoLocation, serverC2, SerType::BINARY)) {
        std::cerr << " Error writing ciphertext 2" << std::endl;
    }

    if (!Serial::SerializeToFile(DATAFOLDER + cipherThreeLocation, serverC3, SerType::BINARY)) {
        std::cerr << " Error writing ciphertext 3" << std::endl;
    }
    std::cout << "Serialization finished" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "This program requres the subdirectory `" << DATAFOLDER << "' to exist, otherwise you will get "
              << "an error writing serializations." << std::endl;
    
    std::string securityLevel;
    uint32_t N, depth, batchSize;
    bool generateBootstrapKey = false;

    if (argc == 6) {
        securityLevel = argv[1];
        N = static_cast<uint32_t>(std::pow(2, std::stoi(argv[2])));
        batchSize = static_cast<uint32_t>(std::pow(2, std::stoi(argv[3])));

        if (securityLevel == "none" && std::stoi(argv[3]) >= std::stoi(argv[2])){
            std::cerr << "Batch size must be less than the ring dimension" << std::endl;
            std::exit(1);
        }

        depth = std::stoul(argv[4]);
        generateBootstrapKey = to_lower(std::string(argv[5])) == "true";

        std::cout << "securityLevel: " << securityLevel << std::endl;
        std::cout << "N: " << N << std::endl;
        std::cout << "batchSize: " << batchSize << std::endl;
        std::cout << "depth: " << depth << std::endl;
    } else {
        std::cout << "Invalid number of parameters used. Using default values.\n";
        // Default values if the number of parameters passed is incorrect.
        securityLevel = "128c";
        N = static_cast<uint32_t>(std::pow(2, 12));
        batchSize = 4096;
        depth = 5;
        generateBootstrapKey = false;
    }

    fileSetupAndWrite(securityLevel, N, batchSize, depth, generateBootstrapKey);
}
