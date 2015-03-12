/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#include "itkBSplineDeformableTransform.h"         // original ITK
#include "itkAdvancedBSplineDeformableTransform.h" // original elastix
#include "itkRecursiveBSplineTransform.h"          // recursive version
//#define LINKINGERRORSFIXED
#ifdef LINKINGERRORSFIXED
#include "itkRecursivePermutedBSplineTransform.h"          // recursive version with permuted parameter order (first xyz, then spatial dimensions)
#endif
//#include "itkBSplineTransform.h" // new ITK4

#include "itkGridScheduleComputer.h"

#include "itkTimeProbe.h"
#include "itkTimeProbesCollectorBase.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"
#include <fstream>
#include <iomanip>

//-------------------------------------------------------------------------------------

int
main( int argc, char * argv[] )
{
  /** Some basic type definitions.
   * NOTE: don't change the dimension or the spline order, since the
   * hard-coded ground truth depends on this.
   */
  const unsigned int Dimension   = 3;
  const unsigned int SplineOrder = 3;
  typedef double CoordinateRepresentationType;
  //const double distance = 1e-3; // the allowable distance
  //const double allowedTimeDifference = 0.1; // 10% is considered within limits

  /** The number of calls to Evaluate(). Distinguish between
   * Debug and Release mode.
   */
#ifndef NDEBUG
  //unsigned int N = static_cast< unsigned int >( 1e3 );
  unsigned int N = static_cast< unsigned int >( 1e3 );
#elif REDUCEDTEST
  unsigned int N = static_cast< unsigned int >( 1e5 );
#else
  unsigned int N = static_cast< unsigned int >( 1e6 );
#endif
  std::cerr << "N = " << N << std::endl;

  /** Check. */
  if( argc != 3 )
  {
    std::cerr << "ERROR: You should specify two text files with the B-spline "
              << "transformation parameters." << std::endl;
    return 1;
  }

  /** Other typedefs. */
  typedef itk::BSplineDeformableTransform<
    CoordinateRepresentationType, Dimension, SplineOrder >    ITKTransformType;
  typedef itk::AdvancedBSplineDeformableTransform<
    CoordinateRepresentationType, Dimension, SplineOrder >    TransformType;
  typedef itk::RecursiveBSplineTransform<
    CoordinateRepresentationType, Dimension, SplineOrder >    RecursiveTransformType;
#ifdef LINKINGERRORSFIXED
  typedef itk::RecursivePermutedBSplineTransform<
    CoordinateRepresentationType, Dimension, SplineOrder >    RecursivePermutedTransformType;
#endif
  typedef TransformType::JacobianType                  JacobianType;
  typedef TransformType::SpatialJacobianType           SpatialJacobianType;
  typedef TransformType::SpatialHessianType            SpatialHessianType;
  typedef TransformType::JacobianOfSpatialJacobianType JacobianOfSpatialJacobianType;
  typedef TransformType::JacobianOfSpatialHessianType  JacobianOfSpatialHessianType;
  typedef TransformType::NonZeroJacobianIndicesType    NonZeroJacobianIndicesType;
  typedef TransformType::NumberOfParametersType        NumberOfParametersType;
  typedef TransformType::InputPointType                InputPointType;
  typedef TransformType::OutputPointType               OutputPointType;
  typedef TransformType::ParametersType                ParametersType;
  typedef TransformType::ImagePointer     CoefficientImagePointer;
  typedef itk::Image< CoordinateRepresentationType,
    Dimension >                                         InputImageType;
  typedef InputImageType::RegionType    RegionType;
  typedef InputImageType::SizeType      SizeType;
  typedef InputImageType::IndexType     IndexType;
  typedef InputImageType::SpacingType   SpacingType;
  typedef InputImageType::PointType     OriginType;
  typedef InputImageType::DirectionType DirectionType;
  typedef itk::Statistics::MersenneTwisterRandomVariateGenerator MersenneTwisterType;

  /** Create the transforms. */
  ITKTransformType::Pointer transformITK = ITKTransformType::New();
  TransformType::Pointer    transform    = TransformType::New();
  RecursiveTransformType::Pointer recursiveTransform    = RecursiveTransformType::New();
#ifdef LINKINGERRORSFIXED
  RecursivePermutedTransformType::Pointer recursivePermutedTransform    = RecursivePermutedTransformType::New();
#endif

  /** Setup the B-spline transform:
   * (GridSize 44 43 35)
   * (GridIndex 0 0 0)
   * (GridSpacing 10.7832773148 11.2116431394 11.8648235177)
   * (GridOrigin -237.6759555555 -239.9488431747 -344.2315805162)
   */
  std::ifstream  input( argv[ 1 ] );
  if( !input.is_open() )
  {
    std::cerr << "ERROR: could not open the text file containing the "
              << "parameter values." << std::endl;
    return 1;
  }
  int dimsInPar1;
  input >> dimsInPar1;
  if (dimsInPar1 != Dimension ) {
    std::cerr << "ERROR: The file containing the parameters specifies " << dimsInPar1
          << " dimensions, while this test is compiled for " << Dimension
              << "dimensions." << std::endl;
    return 1;
  }

  SizeType gridSize;
  for (unsigned int i = 0; i < Dimension ; ++i  ) {
    input >> gridSize[ i ] ;
    std::cerr << "Gridsize dimension " << i << " = " << gridSize[ i ]  << std::endl;
  }
  //gridSize[ 0 ] = 44; gridSize[ 1 ] = 43; gridSize[ 2 ] = 35;
  //gridSize[ 0 ] = 68; gridSize[ 1 ] = 69; gridSize[ 2 ] = 64;
  IndexType gridIndex;
  gridIndex.Fill( 0 );
  RegionType gridRegion;
  gridRegion.SetSize( gridSize );
  gridRegion.SetIndex( gridIndex );
  SpacingType gridSpacing;
  gridSpacing[ 0 ] = 10.7832773148;
  gridSpacing[ 1 ] = 11.2116431394;
  gridSpacing[ 2 ] = 11.8648235177;
  OriginType gridOrigin;
  gridOrigin[ 0 ] = -237.6759555555;
  gridOrigin[ 1 ] = -239.9488431747;
  gridOrigin[ 2 ] = -344.2315805162;
  DirectionType gridDirection;
  gridDirection.SetIdentity();

  transformITK->SetGridOrigin( gridOrigin );
  transformITK->SetGridSpacing( gridSpacing );
  transformITK->SetGridRegion( gridRegion );
  transformITK->SetGridDirection( gridDirection );

  transform->SetGridOrigin( gridOrigin );
  transform->SetGridSpacing( gridSpacing );
  transform->SetGridRegion( gridRegion );
  transform->SetGridDirection( gridDirection );

  recursiveTransform->SetGridOrigin( gridOrigin );
  recursiveTransform->SetGridSpacing( gridSpacing );
  recursiveTransform->SetGridRegion( gridRegion );
  recursiveTransform->SetGridDirection( gridDirection );

#ifdef LINKINGERRORSFIXED
  recursivePermutedTransform->SetGridOrigin( gridOrigin );
  recursivePermutedTransform->SetGridSpacing( gridSpacing );
  recursivePermutedTransform->SetGridRegion( gridRegion );
  recursivePermutedTransform->SetGridDirection( gridDirection );
#endif
  //ParametersType fixPar( Dimension * ( 3 + Dimension ) );
  //fixPar[ 0 ] = gridSize[ 0 ]; fixPar[ 1 ] = gridSize[ 1 ]; fixPar[ 2 ] = gridSize[ 2 ];
  //fixPar[ 3 ] = gridOrigin[ 0 ]; fixPar[ 4 ] = gridOrigin[ 1 ]; fixPar[ 5 ] = gridOrigin[ 2 ];
  //fixPar[ 6 ] = gridSpacing[ 0 ]; fixPar[ 7 ] = gridSpacing[ 1 ]; fixPar[ 8 ] = gridSpacing[ 2 ];
  //fixPar[ 9 ] = gridDirection[ 0 ][ 0 ]; fixPar[ 10 ] = gridDirection[ 0 ][ 1 ]; fixPar[ 11 ] = gridDirection[ 0 ][ 2 ];
  //fixPar[ 12 ] = gridDirection[ 1 ][ 0 ]; fixPar[ 13 ] = gridDirection[ 1 ][ 1 ]; fixPar[ 14 ] = gridDirection[ 1 ][ 2 ];
  //fixPar[ 15 ] = gridDirection[ 2 ][ 0 ]; fixPar[ 16 ] = gridDirection[ 2 ][ 1 ]; fixPar[ 17 ] = gridDirection[ 2 ][ 2 ];
  //transformITK->SetFixedParameters( fixPar );

  /** Now read the parameters as defined in the file par.txt. */
  std::cerr << "Loading parameters from file 1";
  ParametersType parameters( transform->GetNumberOfParameters() );

  for( unsigned int i = 0; i < parameters.GetSize(); ++i )
  {
    input >> parameters[ i ];
  }
  transformITK->SetParameters( parameters );
  transform->SetParameters( parameters );
  std::cerr << ", from file 2 ";

  ParametersType parameters2( transform->GetNumberOfParameters() );
  std::ifstream  input2( argv[ 2 ] );
  if( !input2.is_open() )
  {
    std::cerr << "ERROR: could not open the text file containing the "
              << "parameter2 values." << std::endl;
    return 1;
  }
  int dimsInPar2;
  input2 >> dimsInPar2;
  if (dimsInPar2 != Dimension ) {
    std::cerr << "ERROR: The second file containing the parameters specifies " << dimsInPar2
          << " dimensions, while this test is compiled for " << Dimension
              << "dimensions." << std::endl;
    return 1;
  }

  for (unsigned int i = 0; i < Dimension ; ++i  ) {
    int temp;
    input2 >> temp;
    if (temp != gridSize[ i ]  ) {
      std::cerr << "ERROR: The second file containing the parameters differs in gridsize from the first file." << std::endl;
      return 1;
    }
  }
  for( unsigned int i = 0; i < parameters2.GetSize(); ++i )
  {
    input2 >> parameters2[ i ];
  }
#ifdef LINKINGERRORSFIXED
  recursivePermutedTransform->SetParameters( parameters2 );
#endif
  recursiveTransform->SetParameters( parameters );
  std::cerr <<  " - done"<< std::endl;

  /** Get the number of nonzero Jacobian indices. */
  const NumberOfParametersType nonzji = transform->GetNumberOfNonZeroJacobianIndices();

  /** Declare variables. */
  InputPointType inputPoint;
  inputPoint.Fill( 4.1 );
  JacobianType                  jacobian;
  SpatialJacobianType           spatialJacobian;
  SpatialHessianType            spatialHessian;
  JacobianOfSpatialJacobianType jacobianOfSpatialJacobian;
  JacobianOfSpatialHessianType  jacobianOfSpatialHessian;
  NonZeroJacobianIndicesType    nzji, nzjiElastix, nzjiRecursive;

  /** Resize some of the variables. */
  nzji.resize( nonzji );
  jacobian.SetSize( Dimension, nonzji );
  jacobianOfSpatialJacobian.resize( nonzji );
  jacobianOfSpatialHessian.resize( nonzji );
  jacobian.Fill( 0.0 );

  /**
   *
   * Call functions for testing that they don't crash.
   *
   */

  /** The transform point. */
  recursiveTransform->TransformPoint( inputPoint );
#ifdef LINKINGERRORSFIXED
  recursivePermutedTransform->TransformPoint( inputPoint );
#endif

  /** The Jacobian. *
  recursiveTransform->GetJacobian( inputPoint, jacobian, nzji );

  /** The spatial Jacobian. */
  //recursiveTransform->GetSpatialJacobian( inputPoint, spatialJacobian );
  // crashes

  /** The spatial Hessian. *
  recursiveTransform->GetSpatialHessian( inputPoint, spatialHessian );

  /** The Jacobian of the spatial Jacobian. *
  recursiveTransform->GetJacobianOfSpatialJacobian( inputPoint,
    jacobianOfSpatialJacobian, nzji );

  recursiveTransform->GetJacobianOfSpatialJacobian( inputPoint,
    spatialJacobian, jacobianOfSpatialJacobian, nzji );

  /** The Jacobian of the spatial Hessian. *
  recursiveTransform->GetJacobianOfSpatialHessian( inputPoint,
    jacobianOfSpatialHessian, nzji );

  recursiveTransform->GetJacobianOfSpatialHessian( inputPoint,
    spatialHessian, jacobianOfSpatialHessian, nzji );

  /**
   *
   * Test timing
   *
   */

  itk::TimeProbesCollectorBase timeCollector;
  OutputPointType opp;
  TransformType::WeightsType                weights;
  RecursiveTransformType::WeightsType       weights2;
  TransformType::ParameterIndexArrayType    indices;
  RecursiveTransformType::ParameterIndexArrayType indices2;

  const unsigned int dummyNum = vcl_pow( static_cast< double >( SplineOrder + 1 ),static_cast< double >( Dimension ) );
  weights.SetSize( dummyNum );
  indices.SetSize( dummyNum );
  weights2.SetSize( dummyNum );
  indices2.SetSize( dummyNum );

  bool isInside = true;

  // Generate a list of random points
  MersenneTwisterType::Pointer mersenneTwister = MersenneTwisterType::New();
  mersenneTwister->Initialize( 140377 );
  std::vector< InputPointType > pointList( N );
  std::vector< OutputPointType > transformedPointList1( N );
  std::vector< OutputPointType > transformedPointList2( N );
  std::vector< OutputPointType > transformedPointList3( N );
  std::vector< OutputPointType > transformedPointList4( N );
  std::vector< OutputPointType > transformedPointList5( N );
  std::vector< OutputPointType > transformedPointList6( N );

  IndexType dummyIndex;
  CoefficientImagePointer coefficientImage = transform->GetCoefficientImages()[0];
  for( unsigned int i = 0; i < N; ++i )
  {
    for( unsigned int j = 0; j < Dimension; ++j )
    {
      dummyIndex[ j ] = mersenneTwister->GetUniformVariate( 2, gridSize[ j ] - 3 );
    }
    coefficientImage->TransformIndexToPhysicalPoint( dummyIndex, pointList[ i ] );
  }

  /** Time the implementation of the TransformPoint. */
  timeCollector.Start( "TransformPoint elastix          " );
  for( unsigned int i = 0; i < N; ++i )
  {
    transformedPointList1[i] = transform->TransformPoint( pointList[ i ] );
    //transform->TransformPoint( pointList[ i ], opp, weights, indices, isInside );

    if( isInside == false ){ printf("error point is not in the image"); } // Just to make sure the previous is not optimized away
  }
  timeCollector.Stop(  "TransformPoint elastix          " );

  timeCollector.Start( "TransformPoint recursive        " );
  for( unsigned int i = 0; i < N; ++i )
  {
    transformedPointList2[i] = recursiveTransform->TransformPointOld( pointList[ i ] );
    //recursiveTransform->TransformPoint( pointList[ i ], opp, weights2, indices2, isInside );

    if( isInside == false ){ printf("error point is not in the image"); } // Just to make sure the previous is not optimized away
  }
  timeCollector.Stop(  "TransformPoint recursive        " );

  timeCollector.Start( "TransformPoint recursive vector " );
  for( unsigned int i = 0; i < N; ++i )
  {
    transformedPointList3[i] = recursiveTransform->TransformPoint( pointList[ i ] );
    //recursiveTransform->TransformPointVector( pointList[ i ], opp, weights2, indices2, isInside );

    if( isInside == false ){ printf("error point is not in the image"); } // Just to make sure the previous is not optimized away
  }
  timeCollector.Stop(  "TransformPoint recursive vector " );

  timeCollector.Start( "TransformPoints recursive vector" );
    recursiveTransform->TransformPoints( pointList,  transformedPointList4);
    //recursiveTransform->TransformPointVector( pointList[ i ], opp, weights2, indices2, isInside );
  timeCollector.Stop(  "TransformPoints recursive vector" );

#ifdef LINKINGERRORSFIXED
  timeCollector.Start( "TransformPoint rec.Perm. vector " );
  for( unsigned int i = 0; i < N; ++i )
  {
    transformedPointList5[i] = recursivePermutedTransform->TransformPoint( pointList[ i ] );

    if( isInside == false ){ printf("error point is not in the image"); } // Just to make sure the previous is not optimized away
  }
  timeCollector.Stop(  "TransformPoint rec.Perm. vector " );

  timeCollector.Start( "TransformPoints rec.Perm. vector" );
  recursivePermutedTransform->TransformPoints( pointList,  transformedPointList6);
  timeCollector.Stop(  "TransformPoints rec.Perm. vector" );
#endif

  /** Time the implementation of the Jacobian. */
  timeCollector.Start( "Jacobian elastix                " );
  for( unsigned int i = 0; i < N; ++i )
  {
    transform->GetJacobian( pointList[ i ] , jacobian, nzji );
  }
  timeCollector.Stop(  "Jacobian elastix                " );

  timeCollector.Start( "Jacobian recursive              " );
  for( unsigned int i = 0; i < N; ++i )
  {
    recursiveTransform->GetJacobian( pointList[ i ] , jacobian, nzji );
  }
  timeCollector.Stop(  "Jacobian recursive              " );

  /** Time the implementation of the spatial Jacobian. */
  SpatialJacobianType sj, sjRecursive;
  timeCollector.Start( "SpatialJacobian elastix         " );
  for( unsigned int i = 0; i < N; ++i )
  {
    transform->GetSpatialJacobian( pointList[ i ] , sj );
  }
  timeCollector.Stop(  "SpatialJacobian elastix         " );

  timeCollector.Start( "SpatialJacobian recursive vector" );
  for( unsigned int i = 0; i < N; ++i )
  {
    recursiveTransform->GetSpatialJacobian( pointList[ i ] , sjRecursive );
  }
  timeCollector.Stop(  "SpatialJacobian recursive vector" );

  /** Time the implementation of the spatial Hessian. */
  SpatialHessianType sh, shRecursive;
  timeCollector.Start( "SpatialHessian elastix          " );
  for( unsigned int i = 0; i < N; ++i )
  {
    transform->GetSpatialHessian( pointList[ i ] , sh );
  }
  timeCollector.Stop(  "SpatialHessian elastix          " );

  timeCollector.Start( "SpatialHessian recursive vector " );
  for( unsigned int i = 0; i < N; ++i )
  {
    recursiveTransform->GetSpatialHessian( pointList[ i ] , shRecursive );
  }
  timeCollector.Stop(  "SpatialHessian recursive vector " );

  /** Report. */
  timeCollector.Report();

  /**
   *
   * Test accuracy
   *
   */

  /** These should return the same values as the original ITK functions. */

  /** TransformPoint. */
  OutputPointType opp1, opp2, opp3, opp4, opp5, opp6;
  double differenceNorm1 = 0.0;
  double differenceNorm2 = 0.0;
  double differenceNorm3 = 0.0;
  double differenceNorm4 = 0.0;
  double differenceNorm5 = 0.0;
  double differenceNorm6 = 0.0;
  for( unsigned int i = 0; i < N; ++i )
  {
    opp1 = transformedPointList1[i]; //transform->TransformPoint( pointList[ i ] );
    opp2 = transformedPointList2[i]; //recursiveTransform->TransformPointOld( pointList[ i ] );
    opp3 = transformedPointList3[i]; //recursiveTransform->TransformPoint( pointList[ i ] );
    opp4 = transformedPointList4[i]; // recursiveTransform->TransformPoints
    opp5 = transformedPointList5[i]; // recursivePermutedTransform->TransformPoint( pointList[ i ] );
    opp6 = transformedPointList6[i]; // recursivePermutedTransform->TransformPoints

    for( unsigned int j = 0; j < Dimension; ++j )
    {
      differenceNorm1 += ( opp1[ j ] - opp2[ j ] ) * ( opp1[ j ] - opp2[ j ] );
      differenceNorm2 += ( opp1[ j ] - opp3[ j ] ) * ( opp1[ j ] - opp3[ j ] );
      differenceNorm3 += ( opp2[ j ] - opp3[ j ] ) * ( opp2[ j ] - opp3[ j ] );
      differenceNorm4 += ( opp3[ j ] - opp4[ j ] ) * ( opp3[ j ] - opp4[ j ] );
      differenceNorm5 += ( opp3[ j ] - opp5[ j ] ) * ( opp3[ j ] - opp5[ j ] );
      differenceNorm6 += ( opp3[ j ] - opp6[ j ] ) * ( opp3[ j ] - opp6[ j ] );
    }
  }
  differenceNorm1 = vcl_sqrt( differenceNorm1 ) / N;
  differenceNorm2 = vcl_sqrt( differenceNorm2 ) / N;
  differenceNorm3 = vcl_sqrt( differenceNorm3 ) / N;
  differenceNorm4 = vcl_sqrt( differenceNorm4 ) / N;
  differenceNorm5 = vcl_sqrt( differenceNorm5 ) / N;
  differenceNorm6 = vcl_sqrt( differenceNorm6 ) / N;

  std::cerr << "Recursive B-spline TransformPointOld() MSD with ITK: " << differenceNorm1 << std::endl;
  std::cerr << "Recursive B-spline TransformPoint() MSD with ITK: " << differenceNorm2 << std::endl;
  std::cerr << "Recursive B-spline TransformPoint() MSD with TransformPointOld(): " << differenceNorm3 << std::endl;
  std::cerr << "Recursive B-spline TransformPoint() with TransformPoints(): " << differenceNorm4 << std::endl;
#ifdef LINKINGERRORSFIXED
  std::cerr << "Recursive B-spline TransformPoint() with Permuted TransformPoint(): " << differenceNorm5 << std::endl;
  std::cerr << "Recursive B-spline TransformPoint() with Permuted TransformPoints(): " << differenceNorm6 << std::endl;
#endif
  if( differenceNorm1 > 1e-5 )
  {
    std::cerr << "ERROR: Recursive B-spline TransformPointOld() returning incorrect result." << std::endl;
    return EXIT_FAILURE;
  }
  if( differenceNorm2 > 1e-5 )
  {
    std::cerr << "ERROR: Recursive B-spline TransformPoint() returning incorrect result." << std::endl;
    return EXIT_FAILURE;
  }

  /** Jacobian */
  JacobianType jacobianElastix; jacobianElastix.SetSize( Dimension, nzji.size() ); jacobianElastix.Fill( 0.0 );
  transform->GetJacobian( inputPoint, jacobianElastix, nzjiElastix );

  JacobianType jacobianRecursive; jacobianRecursive.SetSize( Dimension, nzji.size() ); jacobianRecursive.Fill( 0.0 );
  recursiveTransform->GetJacobian( inputPoint, jacobianRecursive, nzjiRecursive );

  JacobianType jacobianDifferenceMatrix = jacobianElastix - jacobianRecursive;
  double jacobianDifference = jacobianDifferenceMatrix.frobenius_norm();
  std::cerr << "The Recursive B-spline GetJacobian() difference is " << jacobianDifference << std::endl;
  if( jacobianDifference > 1e-10 )
  {
    std::cerr << "ERROR: Recursive B-spline GetJacobian() returning incorrect result." << std::endl;
    return EXIT_FAILURE;
  }

  /** NonZeroJacobianIndices. */
  double nzjiDifference = 0.0;
  for( unsigned int i = 0; i < nzjiElastix.size(); ++i )
  {
    nzjiDifference += ( nzjiElastix[i] - nzjiRecursive[i] ) * ( nzjiElastix[i] - nzjiRecursive[i] );
  }
  nzjiDifference = std::sqrt( nzjiDifference );
  std::cerr << "The Recursive B-spline ComputeNonZeroJacobianIndices() difference is " << nzjiDifference << std::endl;
  if( nzjiDifference > 1e-10 )
  {
    std::cerr << "ERROR: Recursive B-spline ComputeNonZeroJacobianIndices() returning incorrect result." << std::endl;
    return EXIT_FAILURE;
  }

  /** Spatial Jacobian */
  transform->GetSpatialJacobian( inputPoint, sj );
  //std::cerr << sj << std::endl;

  recursiveTransform->GetSpatialJacobian( inputPoint, sjRecursive );
  //std::cerr << sjRecursive << std::endl;
  //opp = transform->TransformPoint( inputPoint ) - inputPoint;

  SpatialJacobianType sjDifferenceMatrix = sj - sjRecursive;
  double sjDifference = sjDifferenceMatrix.GetVnlMatrix().frobenius_norm();
  std::cerr << "The Recursive B-spline GetSpatialJacobian() difference is " << sjDifference << std::endl;
  if( sjDifference > 1e-8 )
  {
    std::cerr << "ERROR: Recursive B-spline GetSpatialJacobian() returning incorrect result." << std::endl;
    return EXIT_FAILURE;
  }

  /** Spatial Hessian */
  transform->GetSpatialHessian( inputPoint, sh );
  //std::cerr << sh << std::endl;

  recursiveTransform->GetSpatialHessian( inputPoint, shRecursive );
  //std::cerr << shRecursive << std::endl;

  double shDifference = 0.0;
  for( unsigned int i = 0; i < Dimension; ++i )
  {
    shDifference += ( sh[ i ] - shRecursive[ i ] ).GetVnlMatrix().frobenius_norm();
  }
  std::cerr << "The Recursive B-spline GetSpatialHessian() difference is " << shDifference << std::endl;
  if( shDifference > 1e-8 )
  {
    std::cerr << "ERROR: Recursive B-spline GetSpatialHessian() returning incorrect result." << std::endl;
    return EXIT_FAILURE;
  }

  /** Exercise PrintSelf(). */
  recursiveTransform->Print( std::cerr );

  /** Return a value. */
  return 0;

} // end main