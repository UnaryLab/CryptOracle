#ifndef CKKS_MICROBENCH_RUNNER_H
#define CKKS_MICROBENCH_RUNNER_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <unistd.h> // For getpid()
#include "openfhe.h"
#include <filesystem>
#include <map>
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <linux/hw_breakpoint.h>
#include <sys/syscall.h>
#include <errno.h>

// header files needed for serialization
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"

class BenchmarkRunner {
   private:
    uint32_t functionRepeats, N, Q, qBits, depth, firstModBits, perf_fd = -1;
    std::string securityLevel, profilingFunction;
    bool isPacked, isSetupRun, coldCache;
    unsigned int seed = 1;
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc;
    lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> parameters;
    lbcrypto::KeyPair<lbcrypto::DCRTPoly> keyPair;
    lbcrypto::PublicKey<lbcrypto::DCRTPoly> publicKey;
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> cipherOne, cipherTwo, cipherThree;
    std::unordered_map<std::string, std::function<void()>> operationMap = {
        {"CKKS_GenerateContext", std::bind(&BenchmarkRunner::CKKS_GenerateContext, this)},
        {"CKKS_EvalBootstrapSetup", std::bind(&BenchmarkRunner::CKKS_EvalBootstrapSetup, this)}, // Fails
        {"CKKS_KeyGeneration", std::bind(&BenchmarkRunner::CKKS_KeyGeneration, this)},
        {"CKKS_MultKeyGeneration", std::bind(&BenchmarkRunner::CKKS_MultKeyGeneration, this)},
        {"CKKS_BootstrappingKeyGeneration", std::bind(&BenchmarkRunner::CKKS_BootstrappingKeyGeneration, this)}, // Fails
        {"CKKS_Encryption", std::bind(&BenchmarkRunner::CKKS_Encryption, this)},
        {"CKKS_Decryption", std::bind(&BenchmarkRunner::CKKS_Decryption, this)},
        {"CKKS_Add_Plaintext", std::bind(&BenchmarkRunner::CKKS_Add_Plaintext, this)},
        {"CKKS_Add", std::bind(&BenchmarkRunner::CKKS_Add, this)},
        {"CKKS_Add_Many", std::bind(&BenchmarkRunner::CKKS_Add_Many, this)},
        {"CKKS_Sub", std::bind(&BenchmarkRunner::CKKS_Sub, this)},
        {"CKKS_Sub_Scalar", std::bind(&BenchmarkRunner::CKKS_Sub_Scalar, this)},
        {"CKKS_MultNoRelin", std::bind(&BenchmarkRunner::CKKS_MultNoRelin, this)},
        {"CKKS_Mult", std::bind(&BenchmarkRunner::CKKS_Mult, this)},
        {"CKKS_Mult_Plaintext", std::bind(&BenchmarkRunner::CKKS_Mult_Plaintext, this)},
        {"CKKS_Mult_Scalar", std::bind(&BenchmarkRunner::CKKS_Mult_Scalar, this)},
        {"CKKS_Square", std::bind(&BenchmarkRunner::CKKS_Square, this)},
        {"CKKS_Relin", std::bind(&BenchmarkRunner::CKKS_Relin, this)},
        {"CKKS_Rescale", std::bind(&BenchmarkRunner::CKKS_Rescale, this)},
        {"CKKS_Rotate", std::bind(&BenchmarkRunner::CKKS_Rotate, this)},
        {"CKKS_Fast_Rotate_Precompute", std::bind(&BenchmarkRunner::CKKS_Fast_Rotate_Precompute, this)},
        {"CKKS_Fast_Rotate", std::bind(&BenchmarkRunner::CKKS_Fast_Rotate, this)},
        {"CKKS_Chebyshev_Function", std::bind(&BenchmarkRunner::CKKS_Chebyshev_Function, this)},
        {"CKKS_Chebyshev_Series", std::bind(&BenchmarkRunner::CKKS_Chebyshev_Series, this)},
        {"CKKS_Logistic", std::bind(&BenchmarkRunner::CKKS_Logistic, this)},
        {"CKKS_ModReduce", std::bind(&BenchmarkRunner::CKKS_ModReduce, this)},
        {"CKKS_LevelReduce", std::bind(&BenchmarkRunner::CKKS_LevelReduce, this)},
        {"CKKS_Bootstrap", std::bind(&BenchmarkRunner::CKKS_Bootstrap, this)} // Fails
    };

    std::unordered_map<std::string, lbcrypto::SecurityLevel> securityMap = {
        {"none", lbcrypto::HEStd_NotSet},
        {"128c", lbcrypto::HEStd_128_classic},
        {"192c", lbcrypto::HEStd_192_classic},
        {"256c", lbcrypto::HEStd_256_classic},
        {"128q", lbcrypto::HEStd_128_quantum},
        {"192q", lbcrypto::HEStd_192_quantum},
        {"256q", lbcrypto::HEStd_256_quantum},
    };

    // Save-Load locations for keys
    std::filesystem::path currentPath = std::filesystem::current_path();
    const std::string DATAFOLDER = currentPath.string() + "/util/cryptocontext-generator/serialized-files";
    std::string ccLocation       = "/cryptocontext.txt";
    std::string pubKeyLocation   = "/key_pub.txt";   // Pub key
    std::string multKeyLocation  = "/key_mult.txt";  // relinearization key
    std::string rotKeyLocation   = "/key_rot.txt";   // automorphism / rotation key
    std::string bootstrapKeyLocation   = "/key_bootstrap.txt";   // automorphism / rotation key

    // Save-load locations for RAW ciphertexts
    std::string cipherOneLocation = "/a_input_ciphertext.txt";
    std::string cipherTwoLocation = "/b_input_ciphertext.txt";
    std::string cipherThreeLocation = "/c_input_ciphertext.txt";
    void initializeMaps();

   public:
    BenchmarkRunner(int argc, char* argv[]);

    void benchmarkingSetup();
    void loadContext();
    void runFunction();
    std::vector<double> generateRandomVector(std::size_t slots);
    double generateRandomNumber();
    void CKKS_GenerateContext();
    void CKKS_EvalBootstrapSetup();
    void CKKS_KeyGeneration();
    void CKKS_MultKeyGeneration();
    void CKKS_BootstrappingKeyGeneration();
    void CKKS_Encryption();
    void CKKS_Decryption();
    void CKKS_Add_Plaintext();
    void CKKS_Add();
    void CKKS_Add_Many();
    void CKKS_Sub();
    void CKKS_Sub_Scalar();
    void CKKS_Square();
    void CKKS_MultNoRelin();
    void CKKS_Mult();
    void CKKS_Mult_Plaintext();
    void CKKS_Mult_Scalar();
    void CKKS_Relin();
    void CKKS_Rescale();
    void CKKS_Rotate();
    void CKKS_Fast_Rotate_Precompute();
    void CKKS_Fast_Rotate();
    void CKKS_Chebyshev_Function();
    void CKKS_Chebyshev_Series();
    void CKKS_Logistic();
    void CKKS_ModReduce();
    void CKKS_LevelReduce();
    void CKKS_Bootstrap();

    std::string getProfilingFunction() const;
    int log2IntAsString(const std::string& num) const;
    std::string divideStringBy2(const std::string& num) const;
    std::string to_lower(const std::string& str) const;
};

#endif  // CKKS_MICROBENCH_RUNNER_H