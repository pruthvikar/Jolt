//
//  Double.swift
//  Surge
//
//  Created by Tyler Fleming Cloutier on 1/18/16.
//  Copyright © 2016 Mattt Thompson. All rights reserved.
//

import Foundation
import Accelerate


// MARK: - Double Precision

extension Double: AccelerateFloatingPoint {
    
    // MARK: Sum
    
    public static func sum(_ x: [Double]) -> Double {
        var result: Double = 0.0
        vDSP_sveD(x, 1, &result, vDSP_Length(x.count))
        
        return result
    }
    
    // MARK: Sum of Absolute Values
    
    public static func asum(_ x: [Double]) -> Double {
        return cblas_dasum(Int32(x.count), x, 1)
    }
    
    // MARK: Maximum
    
    public static func max(_ x: [Double]) -> Double {
        var result: Double = 0.0
        vDSP_maxvD(x, 1, &result, vDSP_Length(x.count))
        
        return result
    }
    
    // MARK: Minimum
    
    public static func min(_ x: [Double]) -> Double {
        return cblas_dasum(Int32(x.count), x, 1)
    }
    
    // MARK: Mean
    
    public static func mean(_ x: [Double]) -> Double {
        var result: Double = 0.0
        vDSP_meanvD(x, 1, &result, vDSP_Length(x.count))
        
        return result
    }
    
    // MARK: Mean Magnitude
    
    public static func meamg(_ x: [Double]) -> Double {
        var result: Double = 0.0
        vDSP_meamgvD(x, 1, &result, vDSP_Length(x.count))
        
        return result
    }
    
    // MARK: Mean Square Value
    
    public static func measq(_ x: [Double]) -> Double {
        var result: Double = 0.0
        vDSP_measqvD(x, 1, &result, vDSP_Length(x.count))
        
        return result
    }
    
    // MARK: Add
    
    public static func add(_ x: [Double], y: [Double]) -> [Double] {
        var results = [Double](y)
        cblas_daxpy(Int32(x.count), 1.0, x, 1, &results, 1)
        
        return results
    }
    
    // MARK: Sub
    
    public static func sub(_ x: [Double], y: [Double]) -> [Double] {
        var results = [Double](x)
        cblas_daxpy(Int32(y.count), -1.0, y, 1, &results, 1)
        
        return results
    }
    
    // MARK: Multiply
    
    public static func mul(_ x: [Double], y: [Double]) -> [Double] {
        var results = [Double](repeating: 0.0, count: x.count)
        vDSP_vmulD(x, 1, y, 1, &results, 1, vDSP_Length(x.count))
        
        return results
    }

    public static func mul(_ x: [Double], y: inout Double) -> [Double] {
        var results = [Double](repeating: 0.0, count: x.count)
        vDSP_vsmulD(x, 1, &y, &results, 1, vDSP_Length(x.count))

        return results
    }

    // MARK: Divide
    
    public static func div(_ x: [Double], y: [Double]) -> [Double] {
        var results = [Double](repeating: 0.0, count: x.count)
        vvdiv(&results, x, y, [Int32(x.count)])
        
        return results
    }

  public static func div(_ x: [Double], y: inout Double) -> [Double] {
    var results = [Double](repeating: 0.0, count: x.count)
    vDSP_vsdivD(x, 1, &y, &results, 1, vDSP_Length(x.count))

    return results
  }
    
    // MARK: Modulo
    
    public static func mod(_ x: [Double], y: [Double]) -> [Double] {
        var results = [Double](repeating: 0.0, count: x.count)
        vvfmod(&results, x, y, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Remainder
    
    public static func remainder(_ x: [Double], y: [Double]) -> [Double] {
        var results = [Double](repeating: 0.0, count: x.count)
        vvremainder(&results, x, y, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Square Root
    
    public static func sqrt(_ x: [Double]) -> [Double] {
        var results = [Double](repeating: 0.0, count: x.count)
        vvsqrt(&results, x, [Int32(x.count)])
        
        return results
    }

    // MARK: Square

    public static func square(_ x: [Double]) -> [Double] {
      var results = [Double](repeating: 0.0, count: x.count)
      vDSP_vsqD(x, 1, &results, 1, UInt(x.count))

      return results
    }


    // MARK: Dot Product
    
    public static func dot(_ x: [Double], y: [Double]) -> Double {
        precondition(x.count == y.count, "Vectors must have equal count")
        
        var result: Double = 0.0
        vDSP_dotprD(x, 1, y, 1, &result, vDSP_Length(x.count))
        
        return result
    }
    
    // MARK: Cross Product
    
    public static func cross(_ x: [Double], y: [Double]) -> [Double] {
        precondition(x.count == 3 && y.count == 3, "Cross product vectors must have count 3")
        let m: Matrix<Double> = Matrix(
            [
                [0, -x[2], x[1]],
                [x[2], 0, -x[0]],
                [-x[1], x[0], 0]
            ]
        )
        return (m * Matrix([y])′).grid
    }
    
    // MARK: Vector Norm (These are the same)
    
    public static func norm(_ x: [Double]) -> Double {
        return cblas_dnrm2(Int32(x.count), x, 1)
    }
    
    public static func length(_ x: [Double]) -> Double {
        return cblas_dnrm2(Int32(x.count), x, 1)
        
    }
    
    // MARK: Power
    
    public static func pow(_ x: [Double], y: [Double]) -> [Double] {
        var results = [Double](repeating: 0.0, count: x.count)
        vvpow(&results, x, y, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Absolute Value
    
    public static func abs(_ x: [Double]) -> [Double] {
        var results = [Double](repeating: 0.0, count: x.count)
        vvfabs(&results, x, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Ceiling
    
    public static func ceil(_ x: [Double]) -> [Double] {
        var results = [Double](repeating: 0.0, count: x.count)
        vvceil(&results, x, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Clip
    
    public static func clip(_ x: [Double], low: Double, high: Double) -> [Double] {
        var results = [Double](repeating: 0.0, count: x.count), y = low, z = high
        vDSP_vclipD(x, 1, &y, &z, &results, 1, vDSP_Length(x.count))
        
        return results
    }
    
    // MARK: Copy Sign
    
    public static func copysign(_ magnitude: [Double], sign: [Double]) -> [Double] {
        var results = [Double](repeating: 0.0, count: sign.count)
        vvcopysign(&results, magnitude, sign, [Int32(sign.count)])
        
        return results
    }
    
    // MARK: Floor
    
    public static func floor(_ x: [Double]) -> [Double] {
        var results = [Double](repeating: 0.0, count: x.count)
        vvfloor(&results, x, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Negate
    
    public static func neg(_ x: [Double]) -> [Double] {
        var results = [Double](repeating: 0.0, count: x.count)
        vDSP_vnegD(x, 1, &results, 1, vDSP_Length(x.count))
        
        return results
    }
    
    // MARK: Reciprocal
    
    public static func rec(_ x: [Double]) -> [Double] {
        var results = [Double](repeating: 0.0, count: x.count)
        vvrec(&results, x, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Round
    
    public static func round(_ x: [Double]) -> [Double] {
        var results = [Double](repeating: 0.0, count: x.count)
        vvnint(&results, x, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Threshold
    
    public static func threshold(_ x: [Double], low: Double) -> [Double] {
        var results = [Double](repeating: 0.0, count: x.count), y = low
        vDSP_vthrD(x, 1, &y, &results, 1, vDSP_Length(x.count))
        
        return results
    }
    
    // MARK: Truncate
    
    public static func trunc(_ x: [Double]) -> [Double] {
        var results = [Double](repeating: 0.0, count: x.count)
        vvint(&results, x, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Exponentiation

    public static func exp(_ x: [Double]) -> [Double] {
        var results = [Double](repeating: 0.0, count: x.count)
        vvexp(&results, x, [Int32(x.count)])
        
        return results
    }
    
//    public static func exp(inout x: [Double]) {
//        vvexp(&x, x, [Int32(x.count)])
//    }
    
    // MARK: Square Exponentiation
    
    public static func exp2(_ x: [Double]) -> [Double] {
        var results = [Double](repeating: 0.0, count: x.count)
        vvexp2(&results, x, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Natural Logarithm
    
    public static func log(_ x: [Double]) -> [Double] {
        var results = [Double](x)
        vvlog(&results, x, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Base-2 Logarithm
    
    public static func log2(_ x: [Double]) -> [Double] {
        var results = [Double](x)
        vvlog2(&results, x, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Base-10 Logarithm
    
    public static func log10(_ x: [Double]) -> [Double] {
        var results = [Double](x)
        vvlog10(&results, x, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Logarithmic Exponentiation
    
    public static func logb(_ x: [Double]) -> [Double] {
        var results = [Double](x)
        vvlogb(&results, x, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Fast Fourier Transform
    
    public static func fft(_ input: [Double]) -> [Double] {
        var real = [Double](input)
        var imaginary = [Double](repeating: 0.0, count: input.count)
        var splitComplex = DSPDoubleSplitComplex(realp: &real, imagp: &imaginary)
        
        let length = vDSP_Length(Darwin.floor(Darwin.log2(Double(input.count))))
        let radix = FFTRadix(kFFTRadix2)
        let weights = vDSP_create_fftsetupD(length, radix)
        vDSP_fft_zipD(weights!, &splitComplex, 1, length, FFTDirection(FFT_FORWARD))
        
        var magnitudes = [Double](repeating: 0.0, count: input.count)
        vDSP_zvmagsD(&splitComplex, 1, &magnitudes, 1, vDSP_Length(input.count))
        
        var normalizedMagnitudes = [Double](repeating: 0.0, count: input.count)
        vDSP_vsmulD(sqrt(magnitudes), 1, [2.0 / Double(input.count)], &normalizedMagnitudes, 1, vDSP_Length(input.count))
        
        vDSP_destroy_fftsetupD(weights)
        
        return normalizedMagnitudes
    }
    
    // MARK: Hyperbolic Sine

    public static func sinh(_ x: [Double]) -> [Double] {
        var results = [Double](repeating: 0.0, count: x.count)
        vvsinh(&results, x, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Hyperbolic Cosine

    public static func cosh(_ x: [Double]) -> [Double] {
        var results = [Double](repeating: 0.0, count: x.count)
        vvcosh(&results, x, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Hyperbolic Tangent

    public static func tanh(_ x: [Double]) -> [Double] {
        var results = [Double](repeating: 0.0, count: x.count)
        vvtanh(&results, x, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Inverse Hyperbolic Sine
    
    public static func asinh(_ x: [Double]) -> [Double] {
        var results = [Double](repeating: 0.0, count: x.count)
        vvasinh(&results, x, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Inverse Hyperbolic Cosine
    
    public static func acosh(_ x: [Double]) -> [Double] {
        var results = [Double](repeating: 0.0, count: x.count)
        vvacosh(&results, x, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Inverse Hyperbolic Tangent

    public static func atanh(_ x: [Double]) -> [Double] {
        var results = [Double](repeating: 0.0, count: x.count)
        vvatanh(&results, x, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Sine-Cosine
    
    public static func sincos(_ x: [Double]) -> (sin: [Double], cos: [Double]) {
        var sin = [Double](repeating: 0.0, count: x.count)
        var cos = [Double](repeating: 0.0, count: x.count)
        vvsincos(&sin, &cos, x, [Int32(x.count)])
        
        return (sin, cos)
    }
    
    // MARK: Sine
    
    public static func sin(_ x: [Double]) -> [Double] {
        var results = [Double](repeating: 0.0, count: x.count)
        vvsin(&results, x, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Cosine
    
    public static func cos(_ x: [Double]) -> [Double] {
        var results = [Double](repeating: 0.0, count: x.count)
        vvcos(&results, x, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Tangent
    
    public static func tan(_ x: [Double]) -> [Double] {
        var results = [Double](repeating: 0.0, count: x.count)
        vvtan(&results, x, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Arcsine
    
    public static func asin(_ x: [Double]) -> [Double] {
        var results = [Double](repeating: 0.0, count: x.count)
        vvasin(&results, x, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Arccosine
    
    public static func acos(_ x: [Double]) -> [Double] {
        var results = [Double](repeating: 0.0, count: x.count)
        vvacos(&results, x, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Arctangent

    public static func atan(_ x: [Double]) -> [Double] {
        var results = [Double](repeating: 0.0, count: x.count)
        vvatan(&results, x, [Int32(x.count)])
        
        return results
    }

    public static func atan2(_ x: [Double], y: [Double]) -> [Double] {
        var results = [Double](repeating: 0.0, count: x.count)
        vvatan2(&results, x, y, [Int32(x.count)])

        return results
    }


    // MARK: Radians to Degrees
    
    public static func rad2deg(_ x: [Double]) -> [Double] {
        var results = [Double](repeating: 0.0, count: x.count)
        let divisor = [Double](repeating: M_PI / 180.0, count: x.count)
        vvdiv(&results, x, divisor, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Degrees to Radians
    
    public static func deg2rad(_ x: [Double]) -> [Double] {
        var results = [Double](repeating: 0.0, count: x.count)
        let divisor = [Double](repeating: 180.0 / M_PI, count: x.count)
        vvdiv(&results, x, divisor, [Int32(x.count)])
        
        return results
    }
    
    // MARK: - Matrix Operations
    
    public static func add(_ x: Matrix<Double>, y: Matrix<Double>) -> Matrix<Double> {
        precondition(x.rows == y.rows && x.columns == y.columns, "Matrix dimensions not compatible with addition")
        
        var results = y
        cblas_daxpy(Int32(x.grid.count), 1.0, x.grid, 1, &(results.grid), 1)
        
        return results
    }
    
    public static func add(_ x: Matrix<Double>, alpha: Double) -> Matrix<Double> {
        
        var results = x
        results.grid = [Double](repeating: alpha, count: results.grid.count)
        cblas_daxpy(Int32(x.grid.count), 1.0, x.grid, 1, &(results.grid), 1)
        
        return results
    }
    
    public static func sub(_ x: Matrix<Double>, y: Matrix<Double>) -> Matrix<Double> {
        
        var results = y
        cblas_daxpy(Int32(x.grid.count), -1.0, x.grid, 1, &(results.grid), 1)
        
        return results
    }
    
    public static func sub(_ x: Matrix<Double>, alpha: Double) -> Matrix<Double> {
        
        var results = x
        results.grid = [Double](repeating: alpha, count: results.grid.count)
        cblas_daxpy(Int32(x.grid.count), -1.0, x.grid, 1, &(results.grid), 1)
        
        return results
    }
    
    public static func mul(_ x: Matrix<Double>, y: Matrix<Double>) -> Matrix<Double> {
        precondition(x.columns == y.rows, "Matrix dimensions not compatible with multiplication")
        
        var results = Matrix<Double>(rows: x.rows, columns: y.columns, repeatedValue: 0.0)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(x.rows), Int32(y.columns), Int32(x.columns), 1.0, x.grid, Int32(x.columns), y.grid, Int32(y.columns), 0.0, &(results.grid), Int32(results.columns))
        
        return results
    }
    
    public static func mul(_ x: Matrix<Double>, alpha: Double) -> Matrix<Double> {
        var results = x
        cblas_dscal(Int32(x.grid.count), alpha, &(results.grid), 1)
        
        return results
    }
    
    public static func div(_ x: Matrix<Double>, y: Matrix<Double>) -> Matrix<Double> {
        let yInv = y′
        precondition(x.columns == yInv.rows, "Matrix dimensions not compatible")
        return mul(x, y: yInv)
    }
    
    public static func div(_ x: Matrix<Double>, alpha: Double) -> Matrix<Double> {
        var results = x
        let y = [Double](repeating: alpha, count: x.grid.count)
        vvdiv(&results.grid, x.grid, y, [Int32(x.grid.count)])
        
        return results
    }
    
    public static func inv(_ x : Matrix<Double>) -> Matrix<Double> {
        precondition(x.rows == x.columns, "Matrix must be square")
        
        var results = x
        
        var ipiv = [__CLPK_integer](repeating: 0, count: x.rows * x.rows)
        var lwork = __CLPK_integer(x.columns * x.columns)
        var work = [CDouble](repeating: 0.0, count: Int(lwork))
        var error: __CLPK_integer = 0
        var nc = __CLPK_integer(x.columns)
        
        dgetrf_(&nc, &nc, &(results.grid), &nc, &ipiv, &error)
        dgetri_(&nc, &(results.grid), &nc, &ipiv, &work, &lwork, &error)
        
        assert(error == 0, "Matrix not invertible")
        
        return results
    }
    
    public static func transpose(_ x: Matrix<Double>) -> Matrix<Double> {
        var results = Matrix<Double>(rows: x.columns, columns: x.rows, repeatedValue: 0.0)
        vDSP_mtransD(x.grid, 1, &(results.grid), 1, vDSP_Length(results.rows), vDSP_Length(results.columns))
        
        return results
    }

    
}
