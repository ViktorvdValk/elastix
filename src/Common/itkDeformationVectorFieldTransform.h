/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile$
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkDeformationVectorFieldTransform_H__
#define __itkDeformationVectorFieldTransform_H__

#include "itkBSplineDeformableTransform.h"

namespace itk
{

/** \class DeformationVectorFieldTransform
 * \brief An itk transform based on a DeformationVectorField
 *
 * This class makes it easy to set a deformation vector field
 * as a Transform-object.
 *
 * The class inherits from the 0th-order BSplineDeformableTransform,
 * and converts a VectorImage to the BSpline CoefficientImage.
 * 
 * This is useful if you know for example how to deform each voxel
 * in an image and want to apply it to that image.
 *
 * \ingroup Transforms
 *
 * \todo I take a 0-th order BSplineDeformableTransform, where a
 * 1-st order might be better. However, in that case:
 * take care of image edges, where there is BSpline support!
 */

	template <
    class TScalarType = double,				// Data type for scalars (float or double)
    unsigned int NDimensions = 3 >		// Number of dimensions
		class DeformationVectorFieldTransform:
	public BSplineDeformableTransform< TScalarType, NDimensions, 0 >
	{
	public:
		
		/** Standard class typedefs. */
		typedef DeformationVectorFieldTransform				Self;
		typedef BSplineDeformableTransform<
			TScalarType, NDimensions, 0 >								Superclass;
		typedef SmartPointer< Self >									Pointer;
		typedef SmartPointer< const Self >						ConstPointer;
		
		/** Method for creation through the object factory. */
		itkNewMacro( Self );
		
		/** Run-time type information (and related methods). */
		itkTypeMacro( DeformationVectorFieldTransform, BSplineDeformableTransform );
		
		/** Dimension of the domain space. */
		itkStaticConstMacro( SpaceDimension, unsigned int, NDimensions );
		itkStaticConstMacro( SplineOrder, unsigned int, Superclass::SplineOrder );
		
		/** Typedef's inherited from Superclass. */
		typedef typename Superclass::ScalarType							ScalarType;
		typedef typename Superclass::ParametersType					ParametersType;
		typedef typename Superclass::JacobianType						JacobianType;
		typedef typename Superclass::InputVectorType				InputVectorType;
		typedef typename Superclass::OutputVectorType				OutputVectorType;
		typedef typename Superclass::InputCovariantVectorType		InputCovariantVectorType;
		typedef typename Superclass::OutputCovariantVectorType	OutputCovariantVectorType;
		typedef typename Superclass::InputVnlVectorType			InputVnlVectorType;
		typedef typename Superclass::OutputVnlVectorType		OutputVnlVectorType;
		typedef typename Superclass::InputPointType					InputPointType;
		typedef typename Superclass::OutputPointType				OutputPointType;

		/** Typedef's for BulkTransform. */
		typedef typename Superclass::BulkTransformType			BulkTransformType;
		typedef typename Superclass::BulkTransformPointer		BulkTransformPointer;

		/** Parameters as SpaceDimension number of images. */
    typedef typename Superclass::PixelType              CoefficientPixelType; 
    typedef typename Superclass::ImageType              CoefficientImageType;
    typedef typename Superclass::ImagePointer           CoefficientImagePointer;

		/** Typedef's for VectorImage. */
		typedef Vector< float,
			itkGetStaticConstMacro( SpaceDimension ) >	        CoefficientVectorPixelType;
		typedef Image< CoefficientVectorPixelType,
			itkGetStaticConstMacro( SpaceDimension ) >	        CoefficientVectorImageType;
		typedef typename CoefficientVectorImageType::Pointer	CoefficientVectorImagePointer;
		
		/** Set the coefficient image as a deformation field.
     * The superclass provides a similar function (SetCoeffficientImage),
     * but this function expects an array of nr_of_dim scalar images.
     * The SetCoefficientVectorImage method accepts a VectorImage,
     * which is often more convenient.
     * The method internally just converts this vector image to 
     * nr_of_dim scalar images and passes it on to the 
     * SetCoefficientImage function. */
		virtual void SetCoefficientVectorImage( const CoefficientVectorImageType * vecImage );

		/** Get the coefficient image as a vector image.
     * The vector image is created only on demand. The caller is
     * expected to provide a smart pointer to the resulting image;
     * this stresses the fact that this method does not return a member 
     * variable, like most Get... methods. */
		virtual void GetCoefficientVectorImage( CoefficientVectorImagePointer & vecImage ) const;
		
	protected:
		
		/** The constructor. */
		DeformationVectorFieldTransform();
		/** The destructor. */
		virtual ~DeformationVectorFieldTransform();

	private:
		
		/** The private constructor. */
		DeformationVectorFieldTransform( const Self& );	// purposely not implemented
		/** The private copy constructor. */
		void operator=( const Self& );									// purposely not implemented

		/** Member variables. */
		CoefficientImagePointer m_Images[ SpaceDimension ];
	
	}; // end class DeformationVectorFieldTransform

} // end namespace itk


#ifndef ITK_MANUAL_INSTANTIATION
#include "itkDeformationVectorFieldTransform.txx"
#endif

#endif // end #ifndef __itkDeformationVectorFieldTransform_H__

