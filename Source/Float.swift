//
//  Float.swift
//  Surge
//
//  Created by Tyler Fleming Cloutier on 1/18/16.
//  Copyright © 2016 Mattt Thompson. All rights reserved.
//

import Foundation
import Accelerate

// MARK: - Single Precision

extension Float: AccelerateFloatingPoint {
    
    // MARK: Sum
    
    public static func sum(_ x: [Float]) -> Float {
        var result: Float = 0.0
        vDSP_sve(x, 1, &result, vDSP_Length(x.count))
        
        return result
    }
    
    // MARK: Sum of Absolute Values
    
    public static func asum(_ x: [Float]) -> Float {
        return cblas_sasum(Int32(x.count), x, 1)
    }
    
    // MARK: Maximum
    
    public static func max(_ x: [Float]) -> Float {
        var result: Float = 0.0
        vDSP_maxv(x, 1, &result, vDSP_Length(x.count))
        
        return result
    }
    
    // MARK: Minimum
    
    public static func min(_ x: [Float]) -> Float {
        return cblas_sasum(Int32(x.count), x, 1)
    }
    
    // MARK: Mean
    
    public static func mean(_ x: [Float]) -> Float {
        var result: Float = 0.0
        vDSP_meanv(x, 1, &result, vDSP_Length(x.count))
        
        return result
    }
    
    // MARK: Mean Magnitude
    
    public static func meamg(_ x: [Float]) -> Float {
        var result: Float = 0.0
        vDSP_meamgv(x, 1, &result, vDSP_Length(x.count))
        
        return result    }
    
    // MARK: Mean Square Value
    
    public static func measq(_ x: [Float]) -> Float {
        var result: Float = 0.0
        vDSP_measqv(x, 1, &result, vDSP_Length(x.count))
        
        return result
    }
    
    // MARK: Add
    
    public static func add(_ x: [Float], y: [Float]) -> [Float] {
        var results = [Float](y)
        cblas_saxpy(Int32(x.count), 1.0, x, 1, &results, 1)
        
        return results
    }
    
    // MARK: Sub
    
    public static func sub(_ x: [Float], y: [Float]) -> [Float] {
        var results = [Float](x)
        cblas_saxpy(Int32(y.count), -1.0, y, 1, &results, 1)
        
        return results
    }
    
    // MARK: Multiply
    
    public static func mul(_ x: [Float], y: [Float]) -> [Float] {
        var results = [Float](repeating: 0.0, count: x.count)
        vDSP_vmul(x, 1, y, 1, &results, 1, vDSP_Length(x.count))
        
        return results
    }

    public static func mul(_ x: [Float],y: inout Float) -> [Float] {
      var results = [Float](repeating: 0.0, count: x.count)
      vDSP_vsmul(x, 1, &y, &results, 1, vDSP_Length(x.count))

      return results
    }

    // MARK: Divide
    
    public static func div(_ x: [Float], y: [Float]) -> [Float] {
        var results = [Float](repeating: 0.0, count: x.count)
        vvdivf(&results, x, y, [Int32(x.count)])
        
        return results
    }

  public static func div(_ x: [Float], y: inout Float) -> [Float] {
    var results = [Float](repeating: 0.0, count: x.count)
    vDSP_vsdiv(x, 1, &y, &results, 1, vDSP_Length(x.count))

    return results
  }

    // MARK: Modulo
    
    public static func mod(_ x: [Float], y: [Float]) -> [Float] {
        var results = [Float](repeating: 0.0, count: x.count)
        vvfmodf(&results, x, y, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Remainder
    
    public static func remainder(_ x: [Float], y: [Float]) -> [Float] {
        var results = [Float](repeating: 0.0, count: x.count)
        vvremainderf(&results, x, y, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Square Root
    
    public static func sqrt(_ x: [Float]) -> [Float] {
        var results = [Float](repeating: 0.0, count: x.count)
        vvsqrtf(&results, x, [Int32(x.count)])
        
        return results
    }

      // MARK: Square 
    public static func square(_ x: [Float]) -> [Float] {
        var results = [Float](repeating: 0.0, count: x.count)
        vDSP_vsq(x, 1, &results, 1, UInt(x.count))

        return results
    }


    // MARK: Dot Product
    
    public static func dot(_ x: [Float], y: [Float]) -> Float {
        precondition(x.count == y.count, "Vectors must have equal count")
        
        var result: Float = 0.0
        vDSP_dotpr(x, 1, y, 1, &result, vDSP_Length(x.count))
        
        return result
    }
    
    // MARK: Cross Product
    
    public static func cross(_ x: [Float], y: [Float]) -> [Float] {
        precondition(x.count == 3 && y.count == 3, "Cross product vectors must have count 3")
        let m: Matrix<Float> = Matrix(
            [
                [0, -x[2], x[1]],
                [x[2], 0, -x[0]],
                [-x[1], x[0], 0]
            ]
        )
        return (m * Matrix([y])′).grid
    }
    
    // MARK: Vector Norm (These are the same)
    
    public static func norm(_ x: [Float]) -> Float {
        return cblas_snrm2(Int32(x.count), x, 1)
    }
    
    public static func length(_ x: [Float]) -> Float {
        return cblas_snrm2(Int32(x.count), x, 1)
        
    }
    
    // MARK: Power
    
    public static func pow(_ x: [Float], y: [Float]) -> [Float] {
        var results = [Float](repeating: 0.0, count: x.count)
        vvpowf(&results, x, y, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Absolute Value
    
    public static func abs(_ x: [Float]) -> [Float] {
        var results = [Float](repeating: 0.0, count: x.count)
        vvfabsf(&results, x, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Ceiling
    
    public static func ceil(_ x: [Float]) -> [Float] {
        var results = [Float](repeating: 0.0, count: x.count)
        vvceilf(&results, x, [Int32(x.count)])
        
        return results
    }

    // MARK: Clip
    
    public static func clip(_ x: [Float], low: Float, high: Float) -> [Float] {
        var results = [Float](repeating: 0.0, count: x.count), y = low, z = high
        vDSP_vclip(x, 1, &y, &z, &results, 1, vDSP_Length(x.count))
        
        return results
    }
    
    // MARK: Copy Sign
    
    public static func copysign(_ magnitude: [Float], sign: [Float]) -> [Float] {
        var results = [Float](repeating: 0.0, count: sign.count)
        vvcopysignf(&results, magnitude, sign, [Int32(sign.count)])
        
        return results
    }
    
    // MARK: Floor
    
    public static func floor(_ x: [Float]) -> [Float] {
        var results = [Float](repeating: 0.0, count: x.count)
        vvfloorf(&results, x, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Negate
    
    public static func neg(_ x: [Float]) -> [Float] {
        var results = [Float](repeating: 0.0, count: x.count)
        vDSP_vneg(x, 1, &results, 1, vDSP_Length(x.count))
        
        return results
    }
    
    // MARK: Reciprocal
    
    public static func rec(_ x: [Float]) -> [Float] {
        var results = [Float](repeating: 0.0, count: x.count)
        vvrecf(&results, x, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Round
    
    public static func round(_ x: [Float]) -> [Float] {
        var results = [Float](repeating: 0.0, count: x.count)
        vvnintf(&results, x, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Threshold
    
    public static func threshold(_ x: [Float], low: Float) -> [Float] {
        var results = [Float](repeating: 0.0, count: x.count), y = low
        vDSP_vthr(x, 1, &y, &results, 1, vDSP_Length(x.count))
        
        return results
    }
    
    // MARK: Truncate
    
    public static func trunc(_ x: [Float]) -> [Float] {
        var results = [Float](repeating: 0.0, count: x.count)
        vvintf(&results, x, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Exponentiation
    
    public static func exp(_ x: [Float]) -> [Float] {
        var results = [Float](repeating: 0.0, count: x.count)
        vvexpf(&results, x, [Int32(x.count)])
        
        return results
    }
    
//    public static func exp(inout x: [Float]) {
//        vvexpf(&x, x, [Int32(x.count)])
//    }
    
    // MARK: Square Exponentiation
    
    public static func exp2(_ x: [Float]) -> [Float] {
        var results = [Float](repeating: 0.0, count: x.count)
        vvexp2f(&results, x, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Natural Logarithm
    
    public static func log(_ x: [Float]) -> [Float] {
        var results = [Float](x)
        vvlogf(&results, x, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Base-2 Logarithm
    
    public static func log2(_ x: [Float]) -> [Float] {
        var results = [Float](x)
        vvlog2f(&results, x, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Base-10 Logarithm
    
    public static func log10(_ x: [Float]) -> [Float] {
        var results = [Float](x)
        vvlog10f(&results, x, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Logarithmic Exponentiation
    
    public static func logb(_ x: [Float]) -> [Float] {
        var results = [Float](x)
        vvlogbf(&results, x, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Fast Fourier Transform
    
    public static func fft(_ input: [Float]) -> [Float] {
        var real = [Float](input)
        var imaginary = [Float](repeating: 0.0, count: input.count)
        var splitComplex = DSPSplitComplex(realp: &real, imagp: &imaginary)
        
        let length = vDSP_Length(Darwin.floor(Darwin.log2(Float(input.count))))
        let radix = FFTRadix(kFFTRadix2)
        let weights = vDSP_create_fftsetup(length, radix)
        vDSP_fft_zip(weights!, &splitComplex, 1, length, FFTDirection(FFT_FORWARD))
        
        var magnitudes = [Float](repeating: 0.0, count: input.count)
        vDSP_zvmags(&splitComplex, 1, &magnitudes, 1, vDSP_Length(input.count))
        
        var normalizedMagnitudes = [Float](repeating: 0.0, count: input.count)
        vDSP_vsmul(sqrt(magnitudes), 1, [2.0 / Float(input.count)], &normalizedMagnitudes, 1, vDSP_Length(input.count))
        
        vDSP_destroy_fftsetup(weights)
        
        return normalizedMagnitudes
    }
    
    
    // MARK: Hyperbolic Sine
    
    public static func sinh(_ x: [Float]) -> [Float] {
        var results = [Float](repeating: 0.0, count: x.count)
        vvsinhf(&results, x, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Hyperbolic Cosine
    
    public static func cosh(_ x: [Float]) -> [Float] {
        var results = [Float](repeating: 0.0, count: x.count)
        vvcoshf(&results, x, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Hyperbolic Tangent
    
    public static func tanh(_ x: [Float]) -> [Float] {
        var results = [Float](repeating: 0.0, count: x.count)
        vvtanhf(&results, x, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Inverse Hyperbolic Sine
    
    public static func asinh(_ x: [Float]) -> [Float] {
        var results = [Float](repeating: 0.0, count: x.count)
        vvasinhf(&results, x, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Inverse Hyperbolic Cosine
    
    public static func acosh(_ x: [Float]) -> [Float] {
        var results = [Float](repeating: 0.0, count: x.count)
        vvacoshf(&results, x, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Inverse Hyperbolic Tangent
    
    public static func atanh(_ x: [Float]) -> [Float] {
        var results = [Float](repeating: 0.0, count: x.count)
        vvatanhf(&results, x, [Int32(x.count)])
        
        return results
    }

    public static func atan2(_ x: [Float], y: [Float]) -> [Float] {
        var results = [Float](repeating: 0.0, count: x.count)
        vvatan2f(&results, x, y, [Int32(x.count)])

        return results
    }

    // MARK: Sine-Cosine
    
    public static func sincos(_ x: [Float]) -> (sin: [Float], cos: [Float]) {
        var sin = [Float](repeating: 0.0, count: x.count)
        var cos = [Float](repeating: 0.0, count: x.count)
        vvsincosf(&sin, &cos, x, [Int32(x.count)])
        
        return (sin, cos)
    }
    
    // MARK: Sine
    
    public static func sin(_ x: [Float]) -> [Float] {
        var results = [Float](repeating: 0.0, count: x.count)
        vvsinf(&results, x, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Cosine
    
    public static func cos(_ x: [Float]) -> [Float] {
        var results = [Float](repeating: 0.0, count: x.count)
        vvcosf(&results, x, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Tangent
    
    public static func tan(_ x: [Float]) -> [Float] {
        var results = [Float](repeating: 0.0, count: x.count)
        vvtanf(&results, x, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Arcsine
    
    public static func asin(_ x: [Float]) -> [Float] {
        var results = [Float](repeating: 0.0, count: x.count)
        vvasinf(&results, x, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Arccosine
    
    public static func acos(_ x: [Float]) -> [Float] {
        var results = [Float](repeating: 0.0, count: x.count)
        vvacosf(&results, x, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Arctangent
    
    public static func atan(_ x: [Float]) -> [Float] {
        var results = [Float](repeating: 0.0, count: x.count)
        vvatanf(&results, x, [Int32(x.count)])
        
        return results
    }
    
    
    // MARK: Radians to Degrees
    
    public static func rad2deg(_ x: [Float]) -> [Float] {
        var results = [Float](repeating: 0.0, count: x.count)
        let divisor = [Float](repeating: Float(M_PI / 180.0), count: x.count)
        vvdivf(&results, x, divisor, [Int32(x.count)])
        
        return results
    }
    
    // MARK: Degrees to Radians
    
    public static func deg2rad(_ x: [Float]) -> [Float] {
        var results = [Float](repeating: 0.0, count: x.count)
        let divisor = [Float](repeating: Float(180.0 / M_PI), count: x.count)
        vvdivf(&results, x, divisor, [Int32(x.count)])
        
        return results
    }
    
    // MARK: - Matrix Operations
    
    public static func add(_ x: Matrix<Float>, y: Matrix<Float>) -> Matrix<Float> {
        precondition(x.rows == y.rows && x.columns == y.columns, "Matrix dimensions not compatible with addition")
        
        var results = y
        cblas_saxpy(Int32(x.grid.count), 1.0, x.grid, 1, &(results.grid), 1)
        
        return results
    }
    
    public static func add(_ x: Matrix<Float>, alpha: Float) -> Matrix<Float> {
        
        var results = x
        results.grid = [Float](repeating: alpha, count: results.grid.count)
        cblas_saxpy(Int32(x.grid.count), 1.0, x.grid, 1, &(results.grid), 1)
        
        return results
    }
    
    public static func sub(_ x: Matrix<Float>, y: Matrix<Float>) -> Matrix<Float> {
        
        var results = y
        cblas_saxpy(Int32(x.grid.count), -1.0, x.grid, 1, &(results.grid), 1)
        
        return results
    }
    
    public static func sub(_ x: Matrix<Float>, alpha: Float) -> Matrix<Float> {
        
        var results = x
        results.grid = [Float](repeating: alpha, count: results.grid.count)
        cblas_saxpy(Int32(x.grid.count), -1.0, x.grid, 1, &(results.grid), 1)
        
        return results
    }
    
    public static func mul(_ x: Matrix<Float>, y: Matrix<Float>) -> Matrix<Float> {
        precondition(x.columns == y.rows, "Matrix dimensions not compatible with multiplication")
        
        var results = Matrix<Float>(rows: x.rows, columns: y.columns, repeatedValue: 0.0)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(x.rows), Int32(y.columns), Int32(x.columns), 1.0, x.grid, Int32(x.columns), y.grid, Int32(y.columns), 0.0, &(results.grid), Int32(results.columns))
        
        return results
    }
    
    public static func mul(_ x: Matrix<Float>, alpha: Float) -> Matrix<Float> {
        var results = x
        cblas_sscal(Int32(x.grid.count), alpha, &(results.grid), 1)
        
        return results
    }
    
    public static func div(_ x: Matrix<Float>, y: Matrix<Float>) -> Matrix<Float> {
        let yInv = y′
        precondition(x.columns == yInv.rows, "Matrix dimensions not compatible")
        return Float.mul(x, y: yInv)
    }
    
    public static func div(_ x: Matrix<Float>, alpha: Float) -> Matrix<Float> {
        var results = x
        let y = [Float](repeating: alpha, count: x.grid.count)
        vvdivf(&results.grid, x.grid, y, [Int32(x.grid.count)])
        
        return results
    }
    
    public static func inv(_ x : Matrix<Float>) -> Matrix<Float> {
        precondition(x.rows == x.columns, "Matrix must be square")
        
        var results = x
        
        var ipiv = [__CLPK_integer](repeating: 0, count: x.rows * x.rows)
        var lwork = __CLPK_integer(x.columns * x.columns)
        var work = [CFloat](repeating: 0.0, count: Int(lwork))
        var error: __CLPK_integer = 0
        var nc = __CLPK_integer(x.columns)
        
        sgetrf_(&nc, &nc, &(results.grid), &nc, &ipiv, &error)
        sgetri_(&nc, &(results.grid), &nc, &ipiv, &work, &lwork, &error)
        
        assert(error == 0, "Matrix not invertible")
        
        return results
    }
    
    public static func transpose(_ x: Matrix<Float>) -> Matrix<Float> {
        var results = Matrix<Float>(rows: x.columns, columns: x.rows, repeatedValue: 0.0)
        vDSP_mtrans(x.grid, 1, &(results.grid), 1, vDSP_Length(results.rows), vDSP_Length(results.columns))
        
        return results
    }
    
}


