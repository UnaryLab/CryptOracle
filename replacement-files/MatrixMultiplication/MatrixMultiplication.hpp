//#pragma once

#include "polycircuit/IComponent.hpp"

#include "openfhe/pke/cryptocontext.h"

#define USE_FUNCTION_COUNTERS 1
#define REMOVE_INPLACE_FUNCTIONS 1

namespace polycircuit
{

template <typename ElementType>
class MatrixMultiplication final : public IComponent
{
public:
    explicit MatrixMultiplication(const int matrixSize,
                                  lbcrypto::CryptoContext<ElementType>&& cc,
                                  lbcrypto::Ciphertext<ElementType>&& matrixAC,
                                  lbcrypto::Ciphertext<ElementType>&& matrixBC)
        : m_matrixSize(matrixSize)
        , m_cc(std::move(cc))
        , m_MatrixAC(std::move(matrixAC))
        , m_MatrixBC(std::move(matrixBC))
        , setup(std::move(setup))
    { }
    explicit MatrixMultiplication(const int matrixSize,
                                  const lbcrypto::CryptoContext<ElementType>& cc,
                                  const lbcrypto::Ciphertext<ElementType>& matrixAC,
                                  const lbcrypto::Ciphertext<ElementType>& matrixBC,
                                  const bool setup)
        : m_matrixSize(matrixSize)
        , m_cc(cc)
        , m_MatrixAC(matrixAC)
        , m_MatrixBC(matrixBC)
        , setup(setup)
    { }
    MatrixMultiplication(const MatrixMultiplication&) = default;
    MatrixMultiplication(MatrixMultiplication&&) = default;
    MatrixMultiplication& operator=(const MatrixMultiplication&) = default;
    MatrixMultiplication& operator=(MatrixMultiplication&&) = default;

    Ciphertext evaluate() override
    {
        if (setup){
            return Ciphertext(std::move(m_MatrixAC)); // Return the input ciphertext if input phase
        } 
        else {
        #ifdef USE_FUNCTION_COUNTERS
        int evalAddCount = 0;
        int evalRotateCount = 0;
        int evalMultCount = 0;
        int evalPtxtMultCount = 0;
        int evalAddInPlaceCount = 0;
        #endif
        std::vector<double> mask1(2 * m_matrixSize * m_matrixSize, 0);
        std::vector<std::vector<double>> mask;
        mask.push_back(mask1);
        mask.push_back(std::move(mask1));

        std::vector<lbcrypto::Ciphertext<ElementType>> out(32);
        std::vector<lbcrypto::Plaintext> plaintext_mask(2);

        //#pragma omp parallel for collapse(2)
        for (int k = 0; k < 4; k++)
        {
            for (int i = 0; i < m_matrixSize; i++)
            {
                const int t = (k % 2) * (i) + (1 - (k % 2)) * (i * m_matrixSize);
                mask[(k % 2)][t + (k / 2) * m_matrixSize * m_matrixSize] = 1;
            }
        }

        //#pragma omp parallel for
        for (int j = 0; j < 2; j++)
        {
            plaintext_mask[j] = m_cc->MakeCKKSPackedPlaintext(mask[j]);
        }

        //#pragma omp parallel sections
        {
            //#pragma omp section
            {
                #ifdef REMOVE_INPLACE_FUNCTIONS
                    lbcrypto::Ciphertext<ElementType> temp = m_cc->EvalRotate(m_MatrixAC, -m_matrixSize * m_matrixSize + 1);
                    m_MatrixAC = m_cc->EvalAdd(m_MatrixAC, temp);
                #else
                    m_MatrixAC = m_cc->EvalAdd(m_MatrixAC, m_cc->EvalRotate(m_MatrixAC, -m_matrixSize * m_matrixSize + 1));
                #endif

                #ifdef USE_FUNCTION_COUNTERS
                evalAddCount++;
                evalRotateCount++;
                #endif
            }
            //#pragma omp section
            {
                #ifdef REMOVE_INPLACE_FUNCTIONS
                    lbcrypto::Ciphertext<ElementType> temp = m_cc->EvalRotate(m_MatrixBC, -m_matrixSize * m_matrixSize + m_matrixSize);
                    m_MatrixBC = m_cc->EvalAdd(m_MatrixBC, temp);
                #else
                    m_MatrixBC = m_cc->EvalAdd(m_MatrixBC, m_cc->EvalRotate(m_MatrixBC, -m_matrixSize * m_matrixSize + m_matrixSize));
                #endif

                #ifdef USE_FUNCTION_COUNTERS
                    evalAddCount++;
                    evalRotateCount++;
                #endif
            }
        }

        const int m_matrixSize_log = std::log2(m_matrixSize);
        const int m_matrixSizeHalf_log = std::log2(m_matrixSize / 2);

        //#pragma omp parallel for shared(m_MatrixAC, m_MatrixBC, plaintext_mask)
        for (int t = 0; t < (m_matrixSize / 2); t++)
        {
            std::vector<lbcrypto::Ciphertext<ElementType>> ab1(2), ab2(2);
            if (t != 0)
            {
                ab1[0] = m_cc->EvalRotate(m_MatrixAC, 2 * t);
                #ifdef USE_FUNCTION_COUNTERS
                evalRotateCount++;
                #endif
                ab1[1] = m_cc->EvalRotate(m_MatrixBC, 2 * t * m_matrixSize);
                #ifdef USE_FUNCTION_COUNTERS
                evalRotateCount++;
                #endif
            }
            else
            {
                ab1[0] = m_MatrixAC;
                ab1[1] = m_MatrixBC;
            }
            for (int j = 0; j < 2; j++)
            {
                ab1[j] = m_cc->EvalMult(ab1[j], plaintext_mask[j]);
                #ifdef USE_FUNCTION_COUNTERS
                evalPtxtMultCount++;
                #endif
                for (int k = 0; k < m_matrixSize_log; k++)
                {
                    const int l = -1 * std::pow(2, k);
                    ab2[j] = m_cc->EvalRotate(ab1[j], l * std::pow(m_matrixSize, j));
                    #ifdef USE_FUNCTION_COUNTERS
                    evalRotateCount++;
                    #endif
                    ab1[j] = m_cc->EvalAdd(ab1[j], ab2[j]);
                    #ifdef USE_FUNCTION_COUNTERS
                    evalAddCount++;
                    #endif
                }
            }
            out[t] = m_cc->EvalMult(ab1[0], ab1[1]);
            #ifdef USE_FUNCTION_COUNTERS
            evalMultCount++;
            #endif
        }

        for (int i = 1; i <= m_matrixSizeHalf_log; i++)
        {
            //#pragma omp parallel for
            for (int t = 0; t < static_cast<int>(m_matrixSize / std::pow(2, i + 1)); t++)
            {
                m_cc->EvalAddInPlace(out[t], out[t + static_cast<int>(m_matrixSize / std::pow(2, i + 1))]);
                #ifdef USE_FUNCTION_COUNTERS
                evalAddInPlaceCount++;
                #endif
            }
        }
        #ifdef REMOVE_INPLACE_FUNCTIONS
            lbcrypto::Ciphertext<ElementType> temp = m_cc->EvalRotate(out[0], m_matrixSize * m_matrixSize);
            out[0] = m_cc->EvalAdd(out[0], temp);
        #else
            out[0] = m_cc->EvalAdd(out[0], m_cc->EvalRotate(out[0], m_matrixSize * m_matrixSize));
        #endif

        #ifdef USE_FUNCTION_COUNTERS
            evalAddCount++;
            evalRotateCount++;
        #endif

        #ifdef USE_FUNCTION_COUNTERS
        std::cout << "\nOperation Counts in Matrix Multiplication:" << std::endl;
        std::cout << "EvalAdd operations: " << evalAddCount << std::endl;
        std::cout << "EvalRotate operations: " << evalRotateCount << std::endl;
        std::cout << "EvalMult operations: " << evalMultCount << std::endl;
        std::cout << "EvalAddInPlace operations: " << evalAddInPlaceCount << std::endl;
        std::cout << std::endl;
        #endif

        return Ciphertext(std::move(out[0]));
    }
    }

private:
    const int m_matrixSize;
    const bool setup;
    lbcrypto::CryptoContext<ElementType> m_cc;
    lbcrypto::Ciphertext<ElementType> m_MatrixAC;
    lbcrypto::Ciphertext<ElementType> m_MatrixBC;
};

} // polycircuit
