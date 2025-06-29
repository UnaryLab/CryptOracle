#include "ckks-primitives.h"
#include <linux/perf_event.h>
#include <linux/hw_breakpoint.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <errno.h>

int main(int argc, char* argv[]) {

    BenchmarkRunner runner(argc, argv);
    runner.loadContext();
    runner.benchmarkingSetup();
    runner.runFunction();

    return 0;
}

BenchmarkRunner::BenchmarkRunner(int argc, char* argv[]) {
    if (argc == 5) {
        profilingFunction = argv[1];
        isSetupRun = to_lower(std::string(argv[2])) == "true";
        coldCache = to_lower(std::string(argv[3])) == "true";
        functionRepeats = std::stoul(argv[4]);
        // functionRepeats = 1;
    } else {
        std::cout << "Invalid number of parameters used. Using default values.\n";
        // Default values if the number of parameters passed is incorrect.
        profilingFunction = "CKKS_Add";
        isSetupRun = false;
        coldCache = true;
        functionRepeats = 0;
    }
}

void BenchmarkRunner::loadContext() {
    cc->ClearEvalMultKeys();
    cc->ClearEvalAutomorphismKeys();
    lbcrypto::CryptoContextFactory<lbcrypto::DCRTPoly>::ReleaseAllContexts();

    if (!lbcrypto::Serial::DeserializeFromFile(DATAFOLDER + ccLocation, cc, lbcrypto::SerType::BINARY)) {
        std::cerr << "I cannot read serialized data from: " << DATAFOLDER << ccLocation << std::endl;
        std::exit(1);
    }
    std::cout << "Cryptocontext deserialized" << std::endl;

    if (!lbcrypto::Serial::DeserializeFromFile(DATAFOLDER + pubKeyLocation, publicKey, lbcrypto::SerType::BINARY)) {
        std::cerr << "I cannot read serialized data from: " << DATAFOLDER << pubKeyLocation << std::endl;
        std::exit(1);
    }
    std::cout << "Deserialized public key" << std::endl;

    std::ifstream multKeyIStream(DATAFOLDER + multKeyLocation, std::ios::in | std::ios::binary);
    if (!multKeyIStream.is_open()) {
        std::cerr << "Cannot read serialization from " << DATAFOLDER + multKeyLocation << std::endl;
        std::exit(1);
    }
    if (!cc->DeserializeEvalMultKey(multKeyIStream, lbcrypto::SerType::BINARY)) {
        std::cerr << "Could not deserialize mult key file" << std::endl;
        std::exit(1);
    }

    std::cout << "Deserialized mult key" << std::endl;

    std::ifstream rotKeyIStream(DATAFOLDER + rotKeyLocation, std::ios::in | std::ios::binary);
    if (!rotKeyIStream.is_open()) {
        std::cerr << "Cannot read serialization from " << DATAFOLDER + rotKeyLocation << std::endl;
        std::exit(1);
    }
    if (!cc->DeserializeEvalAutomorphismKey(rotKeyIStream, lbcrypto::SerType::BINARY)) {
        std::cerr << "Could not deserialize rotation key file" << std::endl;
        std::exit(1);
    }
    std::cout << "Deserialized rotation key" << std::endl;

    if (profilingFunction == "CKKS_Bootstrap") {
        std::ifstream bootstrapKeyIStream(DATAFOLDER + bootstrapKeyLocation, std::ios::in | std::ios::binary);
        if (!bootstrapKeyIStream.is_open()) {
            std::cerr << "Cannot read serialization from " << DATAFOLDER + bootstrapKeyLocation << std::endl;
            std::exit(1);
        }
        if (!cc->DeserializeEvalAutomorphismKey(bootstrapKeyIStream, lbcrypto::SerType::BINARY)) {
            std::cerr << "Could not deserialize bootstrap key file" << std::endl;
            std::exit(1);
        }
        std::cout << "Deserialized bootstrapping key" << std::endl;
    }

    if (!lbcrypto::Serial::DeserializeFromFile(DATAFOLDER + cipherOneLocation, cipherOne, lbcrypto::SerType::BINARY)) {
        std::cerr << "Cannot read serialization from " << DATAFOLDER + cipherOneLocation << std::endl;
        std::exit(1);
    }
    std::cout << "Deserialized cipherOne" << std::endl;

    if (!lbcrypto::Serial::DeserializeFromFile(DATAFOLDER + cipherTwoLocation, cipherTwo, lbcrypto::SerType::BINARY)) {
        std::cerr << "Cannot read serialization from " << DATAFOLDER + cipherTwoLocation << std::endl;
        std::exit(1);
    }

    std::cout << "Deserialized cipherTwo" << std::endl;

    if (!lbcrypto::Serial::DeserializeFromFile(DATAFOLDER + cipherThreeLocation, cipherThree, lbcrypto::SerType::BINARY)) {
        std::cerr << "Cannot read serialization from " << DATAFOLDER + cipherThreeLocation << std::endl;
        std::exit(1);
    }

    std::cout << "Deserialized cipherThree" << std::endl;
}

void BenchmarkRunner::benchmarkingSetup() {
    std::string shortLabelAlignment = "\t\t\t\t\t\t", labelAlignment = "\t\t\t\t\t", longLabelAlignment = "\t\t\t\t";
    std::cout << "\n============================= PROGRAM PARAMETERS =============================\n" << std::endl;
    std::cout << "FHE Scheme:" << shortLabelAlignment << cc->getSchemeId() << std::endl;
    std::cout << "Profiling Function:" << labelAlignment << profilingFunction << std::endl;
    std::cout << "Ring Dimension (N):" << labelAlignment << cc->GetRingDimension() << std::endl;
    std::cout << "Security Standard:" << labelAlignment << securityLevel << std::endl;
    std::cout << "Ctxt Modulus (Q) Bitwidth:" << longLabelAlignment << (log2IntAsString(cc->GetModulus().ToString())) << std::endl;
    std::cout << "Small Moduli (q) Bitwidth:" << longLabelAlignment << parameters.GetScalingModSize() << std::endl;
    std::cout << "Ciphertext Size:" << labelAlignment << (ceil(cc->GetRingDimension() * (log2IntAsString(cc->GetModulus().ToString()))) / 1024 / 1024 / 8) << " MB" << std::endl;
    std::cout << "Batch Size:" << shortLabelAlignment << cc->GetEncodingParams()->GetBatchSize() << std::endl;
    std::cout << "Depth:" << shortLabelAlignment <<  cipherOne->GetLevel() << std::endl;
    std::cout << "Setup Run:" << shortLabelAlignment << ((isSetupRun) ? "True" : "False") << std::endl;
}

void BenchmarkRunner::runFunction() {
    auto it = operationMap.find(profilingFunction);
    if (it == operationMap.end()) {
        std::cerr << "Function '" << profilingFunction << "' not found!\n";
        return;
    }

    std::vector<double> call_times;
    uint32_t num_runs = 0;
    constexpr uint32_t hot_cache_num_calls = 5; // Number of times to repeat (< 5 has negligible returns in testing)
    constexpr double minimum_runtime = 500; // Minimum runtime in milliseconds
    std::chrono::duration<double, std::milli> total_time(0.0);

    do {
        std::chrono::time_point<std::chrono::high_resolution_clock> start, end;

        if (coldCache) {
            // loadContext(); // Regenerate every time for cold cache
            std::cout << std::endl << "START RUN " << num_runs << std::endl;
            start = std::chrono::high_resolution_clock::now();
            // PerfCounter perf({PerfCounter::INSTRUCTIONS, PerfCounter::CACHE_REFERENCES, PerfCounter::CPU_CYCLES, PerfCounter::BRANCHES, PerfCounter::BRANCH_MISSES});
            it->second(); // Call the function
            end = std::chrono::high_resolution_clock::now();
            std::cout << "END RUN " << num_runs << std::endl;
        } else {
            // Warm the cache first time
            if (num_runs == 0) {
                for (uint32_t i = 0; i < hot_cache_num_calls; ++i)
                    it->second(); // Call the function
            }
            std::cout << std::endl << "START RUN " << num_runs << std::endl;
            start = std::chrono::high_resolution_clock::now();
            it->second(); // Call the function
            end = std::chrono::high_resolution_clock::now();
            std::cout << "END RUN " << num_runs << std::endl << std::endl;
        }

        std::chrono::duration<double, std::milli> duration = end - start;
        total_time += duration;
        call_times.push_back(duration.count());
        num_runs++;
    } while ((functionRepeats != 0 && num_runs < functionRepeats) || (functionRepeats == 0 && total_time.count() < minimum_runtime));

    std::cout << std::endl << profilingFunction << " was run: " << num_runs << " time(s)." << std::endl << std::endl;
    std::cout << "Time taken: " << total_time.count() << " milliseconds" << std::endl;
    std::cout << "Average time taken: " << total_time.count() / num_runs << " milliseconds" << std::endl;
}

std::vector<double> BenchmarkRunner::generateRandomVector(std::size_t slots) {
    std::vector<double> test_vector(slots);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (auto& val : test_vector) {
        val = dis(gen);
    }

    return test_vector;
}

double BenchmarkRunner::generateRandomNumber(){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    return dis(gen);
}

void BenchmarkRunner::CKKS_GenerateContext() {
    // Key distribution parameters
    lbcrypto::SecretKeyDist secretKeyDist = lbcrypto::UNIFORM_TERNARY;
    parameters.SetSecretKeyDist(secretKeyDist);

    // Keyswitching parameters
    usint dnum = 3;
    parameters.SetNumLargeDigits(dnum);

    // Bootstrapping parameters
    std::vector<uint32_t> levelBudget = {3, 3};
    if (N >= 65536 || securityLevel != "none") {
        levelBudget = {4, 4};
    }

    std::vector<uint32_t> bsgsDim = {0, 0};

    // Multiplicative depth parameter
    depth = 10;
    uint32_t multDepth = depth + lbcrypto::FHECKKSRNS::GetBootstrapDepth(levelBudget, secretKeyDist);
    parameters.SetMultiplicativeDepth(multDepth);

    // Security level standards
    lbcrypto::SecurityLevel ckksSecurityLevel;

    if (securityLevel != "none") {
        auto it = securityMap.find(securityLevel);

        if (it != securityMap.end())
            ckksSecurityLevel = it->second;
        else {
            std::cout << "Invalid security standard. Using default value of HEStd_128_classic." << std::endl;
            ckksSecurityLevel = lbcrypto::HEStd_128_classic;
        }
    } else {
        ckksSecurityLevel = lbcrypto::HEStd_NotSet;
        if (Q < 500) {
            std::cout << "Warning! Specified ciphertext modulus incompatible with specified depth. Updating modulus to minimum value" << std::endl;
            Q = 550;
        }

        firstModBits = uint32_t((Q + multDepth) / (multDepth + 1));

        if (firstModBits > 60) {
            std::cout << "Warning! Specified ciphertext modulus incompatible with specified depth. Adjusting modulus to fit depth" << std::endl;
            while (firstModBits > 60) {
                Q -= 10;
                firstModBits = uint32_t((Q + multDepth) / (multDepth + 1));
            }
        } else if (firstModBits < 21) {
            std::cout << "Warning! Specified ciphertext modulus incompatible with specified depth. Adjusting modulus to fit depth" << std::endl;
            while (firstModBits < 21) {
                Q += 10;
                firstModBits = uint32_t((Q + multDepth) / (multDepth + 1));
            }
        }

        lbcrypto::ScalingTechnique rescaleTech = lbcrypto::FIXEDMANUAL;

        parameters.SetScalingTechnique(rescaleTech);
        parameters.SetScalingModSize(firstModBits - 1);
        parameters.SetFirstModSize(firstModBits);
        parameters.SetRingDim(N);  // Set the N value of the ciphertext polynomial
    }

    parameters.SetSecurityLevel(ckksSecurityLevel);

    // Batching Parameters
    if (isPacked && securityLevel == "none")
        parameters.SetBatchSize(parameters.GetRingDim() / 2);
    else if (isPacked)
        parameters.SetBatchSize(parameters.GetRingDim() / 8);
    else
        parameters.SetBatchSize(8);

    std::cout << std::endl << "All crypto parameters: " << std::endl << parameters << std::endl;

    if (!isSetupRun || profilingFunction != "CKKS_GenerateContext"){ // if called from other function, always run this
        cc = GenCryptoContext(parameters);

        cc->Enable(lbcrypto::PKE);
        cc->Enable(lbcrypto::KEYSWITCH);
        cc->Enable(lbcrypto::LEVELEDSHE);
        cc->Enable(lbcrypto::ADVANCEDSHE);
        cc->Enable(lbcrypto::FHE);
    }
}

void BenchmarkRunner::CKKS_EvalBootstrapSetup(){
    std::vector<uint32_t> levelBudget = {3, 3};
    if (N >= 65536 || securityLevel != "none") {
        levelBudget = {4, 4};
    }

    std::vector<uint32_t> bsgsDim = {0, 0};

    if (!isSetupRun)
        cc->EvalBootstrapSetup(levelBudget, bsgsDim, cc->GetEncodingParams()->GetBatchSize());
}

void BenchmarkRunner::CKKS_KeyGeneration() {
    if (!isSetupRun)
        keyPair = cc->KeyGen();
}

void BenchmarkRunner::CKKS_MultKeyGeneration() {
    keyPair = cc->KeyGen();

    if (!isSetupRun)
        cc->EvalMultKeyGen(keyPair.secretKey);
}

void BenchmarkRunner::CKKS_BootstrappingKeyGeneration() {
    keyPair = cc->KeyGen();
    usint slots = cc->GetEncodingParams()->GetBatchSize();

    std::vector<uint32_t> levelBudget = {3, 3};
    if (N >= 65536 || securityLevel != "none") {
        levelBudget = {4, 4};
    }
    std::vector<uint32_t> bsgsDim = {0, 0};
    cc->EvalBootstrapSetup(levelBudget, bsgsDim, slots);

    if (!isSetupRun)
        cc->EvalBootstrapKeyGen(keyPair.secretKey, slots);
}

void BenchmarkRunner::CKKS_Encryption() {
    keyPair = cc->KeyGen();
    usint slots = cc->GetEncodingParams()->GetBatchSize();

    std::vector<double> test_vector = generateRandomVector(slots);

    auto plaintext = cc->MakeCKKSPackedPlaintext(test_vector);
    plaintext->SetLength(slots);

    if (!isSetupRun)
        auto ciphertext = cc->Encrypt(keyPair.publicKey, plaintext);
}

void BenchmarkRunner::CKKS_Decryption() {
    keyPair = cc->KeyGen();
    usint slots = cc->GetEncodingParams()->GetBatchSize();

    std::vector<double> test_vector = generateRandomVector(slots);

    auto plaintext1 = cc->MakeCKKSPackedPlaintext(test_vector);
    plaintext1->SetLength(slots);
    auto cipherOne = cc->Encrypt(keyPair.publicKey, plaintext1);
    cipherOne = cc->LevelReduce(cipherOne, nullptr, 1);

    lbcrypto::Plaintext plaintextDec1;

    if (!isSetupRun)
        cc->Decrypt(keyPair.secretKey, cipherOne, &plaintextDec1);
}

void BenchmarkRunner::CKKS_Add_Plaintext() {
    usint slots = cc->GetEncodingParams()->GetBatchSize();
    std::vector<double> test_vector = generateRandomVector(slots);
    auto plaintextOperand = cc->MakeCKKSPackedPlaintext(test_vector);
    plaintextOperand->SetLength(slots);

    if (!isSetupRun)
        auto ciphertextAdd = cc->EvalAdd(cipherOne, plaintextOperand);
}

void BenchmarkRunner::CKKS_Add() {
    if (!isSetupRun)
        auto result = cc->EvalAdd(cipherOne, cipherTwo);
}

void BenchmarkRunner::CKKS_Add_Many() {
    const int num_adds = 9; // Arbitrary, commonly used in LowMemoryResnet20

    std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> ciphertexts(num_adds);

    // Also arbitrary, used to generate new ciphertexts
    for (int i = 0; i < num_adds; ++i)
        ciphertexts[i] = cc->EvalAdd(cipherOne, cipherTwo);

    if (!isSetupRun)
        auto result = cc->EvalAddMany(ciphertexts);
}

void BenchmarkRunner::CKKS_Sub() {
    if (!isSetupRun)
        auto result = cc->EvalSub(cipherOne, cipherTwo);
}

void BenchmarkRunner::CKKS_Sub_Scalar() {
    if (!isSetupRun)
        auto result = cc->EvalSub(cipherOne, generateRandomNumber());
}

void BenchmarkRunner::CKKS_MultNoRelin() {
    if (!isSetupRun)
        auto result = cc->EvalMultNoRelin(cipherOne, cipherTwo);
}

void BenchmarkRunner::CKKS_Mult() {
    if (!isSetupRun)
        auto result = cc->EvalMult(cipherOne, cipherTwo);
}

void BenchmarkRunner::CKKS_Square() {
    if (!isSetupRun)
        auto result = cc->EvalSquare(cipherOne);
}

void BenchmarkRunner::CKKS_Mult_Plaintext() {
    usint slots = cc->GetEncodingParams()->GetBatchSize();

    std::vector<double> test_vector = generateRandomVector(slots);

    auto plaintextOperand = cc->MakeCKKSPackedPlaintext(test_vector);
    plaintextOperand->SetLength(slots);

    if (!isSetupRun)
        auto result = cc->EvalMult(cipherOne, plaintextOperand);
}

void BenchmarkRunner::CKKS_Mult_Scalar() {
    if (!isSetupRun)
        auto result = cc->EvalMult(cipherOne, generateRandomNumber());
}

void BenchmarkRunner::CKKS_Rotate() {
    if (!isSetupRun)
        auto result = cc->EvalRotate(cipherThree, 1);
        // auto result = cc->EvalRotate(cipherThree, 1024);
}

void BenchmarkRunner::CKKS_Fast_Rotate_Precompute() {
    if (!isSetupRun)
        auto precompute = cc->EvalFastRotationPrecompute(cipherOne);
}

void BenchmarkRunner::CKKS_Fast_Rotate() {
    int padding = 1;
    auto precompute = cc->EvalFastRotationPrecompute(cipherOne);

    if (!isSetupRun)
        auto result = cc->EvalFastRotation(cipherOne, padding, cc->GetCyclotomicOrder(), precompute);
}

void BenchmarkRunner::CKKS_Chebyshev_Series() {

    static std::vector<double> coeffVal({1, 0.6349347497444793, 0.0, -0.207226910968973, 0.0, 0.11926554318627501,
                                        0.0, -0.08013715047239724, 0.0, 0.05757992161586102, 0.0, -0.04280910730211544,
                                        0.0, 0.03243169753850008, 0.0, -0.024837099847818355, 0.0, 0.01914266900321147,
                                        0.0, -0.014810271063030017, 0.0, 0.011484876770357881, 0.0, -0.008918658212582378,
                                        0.0, 0.006931783590193362, 0.0, -0.00539036659534493, 0.0, 0.004193061697765822,
                                        0.0, -0.0032623449132251907, 0.0, 0.0025385228671728505, 0.0, -0.0019754432169236547,
                                        0.0, 0.001537332438236821, 0.0, -0.00119641841400218, 0.0, 0.0009311199764515702,
                                        0.0, -0.0007246568620069462, 0.0, 0.0005639769377601768, 0.0, -0.00043892561744415164,
                                        0.0, 0.00034160150542931166, 0.0, -0.00026585589670811906, 0.0, 0.00020690371892740422,
                                        0.0, -0.00016102095964053078, 0.0, 0.0001253092833997599});


    if (!isSetupRun)
        auto result = cc->EvalChebyshevSeries(cipherThree, coeffVal, -25, 25);
}

void BenchmarkRunner::CKKS_Chebyshev_Function() {
    const int relu_degree = 59;
    const int chebRangeStart = -1, chebRangeEnd = 1;
    double scale = generateRandomNumber();

    if (!isSetupRun)
        auto result = cc->EvalChebyshevFunction([scale](double x) -> double {return (1/scale) * x;}, cipherThree, chebRangeStart, chebRangeEnd, relu_degree);
}

void BenchmarkRunner::CKKS_Logistic() {
    const int chebRangeStart = -16, chebRangeEnd = 16, chebDegree = 59;

    if (!isSetupRun)
        auto result = cc->EvalLogistic(cipherTwo, chebRangeStart, chebRangeEnd, chebDegree);
}

void BenchmarkRunner::CKKS_Relin() {
    auto ciphertextMul = cc->EvalMult(cipherOne, cipherTwo);
    if (!isSetupRun)
        auto ciphertext3 = cc->Relinearize(ciphertextMul);
}

void BenchmarkRunner::CKKS_Rescale() {
    auto ciphertextMul = cc->EvalMult(cipherOne, cipherTwo);
    if (!isSetupRun)
        auto ciphertext3 = cc->ModReduce(ciphertextMul);
}

void BenchmarkRunner::CKKS_ModReduce() {
    auto product = cc->EvalMult(cipherOne, cipherTwo);

    if (!isSetupRun)
        auto result = cc->ModReduce(product);
}

void BenchmarkRunner::CKKS_LevelReduce() {
    auto product = cc->EvalMult(cipherOne, cipherTwo);

    if (!isSetupRun)
        auto result = cc->LevelReduce(product, nullptr);
}

void BenchmarkRunner::CKKS_Bootstrap() {
    // // Set the level budget for bootstrapping
    // std::vector<uint32_t> levelBudget = {3, 3};
    
    // // Adjust level budget based on security level and ring dimension (N)
    // if (N >= 65536 || securityLevel != "none") {
    //     levelBudget = {3, 2}; // Adjust the level budget to avoid exceeding the correction factor
    // }

    // // Initialize BSGS dimensions with better values
    // std::vector<uint32_t> bsgsDim = {2, 2};  // Try non-zero BSGS dimensions for better compatibility

    // // Get batch size from encoding parameters
    // usint batchSize = cc->GetEncodingParams()->GetBatchSize();

    // // Setup the bootstrapping parameters
    // cc->EvalBootstrapSetup(levelBudget, bsgsDim, batchSize);

    // // Key generation
    // keyPair = cc->KeyGen();
    // usint slots = cc->GetEncodingParams()->GetBatchSize(); // Retrieve the number of available slots

    // cc->EvalBootstrapKeyGen(keyPair.secretKey, slots);

    // Perform bootstrapping if this is not just a setup run
    // auto result = cc->EvalMult(cipherOne, cipherTwo);
    std::vector<uint32_t> levelBudget = {3,3};
    std::vector<uint32_t> bsgsDim = {0,0};
    usint batchSize = cc->GetEncodingParams()->GetBatchSize();
    cc->EvalBootstrapSetup(levelBudget, bsgsDim, batchSize, 0, true);

    if (!isSetupRun) {
        // Adjust the EvalBootstrap method to handle correction factor
        auto ciphertextAfterBootstrapping = cc->EvalBootstrap(cipherOne);
    }
}

// Function to compute the base-2 logarithm of a large integer represented as a string.
int BenchmarkRunner::log2IntAsString(const std::string& num) const {
    std::string temp = num; // Make a copy of the input to work with.
    int log2 = 0; // This will store the logarithm value.

    // Keep dividing the number by 2 until it becomes 1.
    // Increment the log2 counter for each division.
    while (temp != "1") {
        temp = divideStringBy2(temp);
        log2++;
    }

    return log2;
}

// Function to divide a large number, represented as a string, by 2.
std::string BenchmarkRunner::divideStringBy2(const std::string& num) const {
    if (num.empty()) {
        // If the input string is empty, return "0" as the division result.
        return "0";
    }

    std::string result; // This will store the result of division.
    int carryOver = 0; // This holds the remainder that is carried over to the next digit.

    for (char digitChar : num) {
        // Convert the current character to an integer and add the carry from the previous division.
        int digit = carryOver * 10 + (digitChar - '0');

        // Compute the quotient of the current digit by 2 and add to the result string.
        result.push_back((digit / 2) + '0');

        // Calculate the remainder to carry over to the next digit.
        carryOver = digit % 2;
    }

    // Remove leading zero from the result if it is not the only character in the string.
    if (result[0] == '0' && result.length() > 1) {
        result.erase(0, 1);
    }

    return result;
}

std::string BenchmarkRunner::to_lower(const std::string& str) const {
    std::string lower_str = str;
    std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return lower_str;
}