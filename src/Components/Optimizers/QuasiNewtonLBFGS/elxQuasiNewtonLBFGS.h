/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxQuasiNewtonLBFGS_h
#define __elxQuasiNewtonLBFGS_h

#include "itkQuasiNewtonLBFGSOptimizer.h"
#include "itkMoreThuenteLineSearchOptimizer.h"
#include "elxIncludes.h"

namespace elastix
{
using namespace itk;


	/**
	 * \class QuasiNewtonLBFGS
	 * \brief An optimizer based on the itk::QuasiNewtonLBFGSOptimizer.
	 *
	 * The QuasiNewtonLBFGS class is a wrap around the QuasiNewtonLBFGSOptimizer.
	 * It uses the itk::MoreThuenteLineSearchOptimizer.
	 * Please read the documentation of these classes to find out more about it.
	 * 
	 * This optimizer supports the NewSamplesEveryIteration option. It requests
	 * new samples for the computation of each search direction (not during
	 * the line search). Actually this makes no sense for a QuasiNewton optimizer.
	 * So, think twice before using the NewSamplesEveryIteration option.
	 *
	 * The parameters used in this class are:
	 * \parameter Optimizer: Select this optimizer as follows:\n
	 *		<tt>(Optimizer "QuasiNewtonLBFGS")</tt>
	 * \parameter GenerateLineSearchIterations: Whether line search iteration
	 *   should be counted as elastix-iterations.\n
	 *   example: <tt>(GenerateLineSearchIterations "true")</tt>\n
	 *   Can only be specified for all resolutions at once. \n
	 *   Default value: "false".\n
	 * \parameter MaximumNumberOfIterations: The maximum number of iterations in each resolution. \n
	 *		example: <tt>(MaximumNumberOfIterations 100 100 50)</tt> \n
	 *    Default value: 100.\n
	 * \parameter MaximumNumberOfLineSearchIterations: The maximum number of iterations in each resolution. \n
	 *		example: <tt>(MaximumNumberOfIterations 10 10 5)</tt> \n
	 *    Default value: 10.\n
	 * \parameter StepLength: Set the length of the initial step tried by the
	 *    itk::MoreThuenteLineSearchOptimizer.\n
	 *		example: <tt>(StepLength 2.0 1.0 0.5)</tt> \n
	 *    Default value: 1.0.\n
	 * \parameter LineSearchValueTolerance: Determine the Wolfe conditions that the
	 *    itk::MoreThuenteLineSearchOptimizer tries to satisfy.\n
	 *		example: <tt>(LineSearchValueTolerance 0.0001 0.0001 0.0001)</tt> \n
	 *    Default value: 0.0001.\n
	 * \parameter LineSearchGradientTolerance: Determine the Wolfe conditions that the
	 *    itk::MoreThuenteLineSearchOptimizer tries to satisfy.\n
	 *		example: <tt>(LineSearchGradientTolerance 0.9 0.9 0.9)</tt> \n
	 *    Default value: 0.9.\n
	 * \parameter GradientMagnitudeTolerance: Stopping criterion. See the documentation of the 
	 *    itk::QuasiNewtonLBFGSOptimizer for more information.\n
	 *		example: <tt>(GradientMagnitudeTolerance 0.001 0.0001 0.000001)</tt> \n
	 *	  Default value: 0.000001.\n
	 * \parameter LBFGSUpdateAccuracy: The "memory" of the optimizer. This determines
	 *    how many past iterations are used to construct the Hessian approximation.
	 *    The higher, the more memory is used, but the better the Hessian approximation.
	 *    If set to zero, The QuasiNewtonLBFGS equals a gradient descent method with
	 *    line search.\n
	 *		example: <tt>(LBFGSUpdateAccuracy 5 10 20)</tt> \n
	 *    Default value: 5.\n
   * \parameter StopIfWolfeNotSatisfied: Whether to stop the optimisation if in one iteration 
	 *    the Wolfe conditions can not be satisfied by the itk::MoreThuenteLineSearchOptimizer.\n
	 *    In general it is wise to do so.\n
	 *    example: <tt>(StopIfWolfeNotSatisfied "true" "false")</tt> \n
	 *    Default value: "true".\n
	 *
	 * \ingroup Optimizers
	 */

	template <class TElastix>
		class QuasiNewtonLBFGS :
		public
			itk::QuasiNewtonLBFGSOptimizer,
		public
			OptimizerBase<TElastix>
	{
	public:

		/** Standard ITK.*/
		typedef QuasiNewtonLBFGS						  			Self;
		typedef QuasiNewtonLBFGSOptimizer						Superclass1;
		typedef OptimizerBase<TElastix>							Superclass2;
		typedef SmartPointer<Self>									Pointer;
		typedef SmartPointer<const Self>						ConstPointer;
		
		/** Method for creation through the object factory. */
		itkNewMacro( Self );
		
		/** Run-time type information (and related methods). */
		itkTypeMacro( QuasiNewtonLBFGS, QuasiNewtonLBFGSOptimizer );
		
		/** Name of this class.
		 * Use this name in the parameter file to select this specific optimizer. \n
		 * example: <tt>(Optimizer "QuasiNewtonLBFGS")</tt>\n
		 */
		elxClassNameMacro( "QuasiNewtonLBFGS" );

		/** Typedef's inherited from Superclass1.*/
	  typedef Superclass1::CostFunctionType								    CostFunctionType;
		typedef Superclass1::CostFunctionPointer						    CostFunctionPointer;
		typedef Superclass1::StopConditionType							    StopConditionType;
		typedef Superclass1::ParametersType									    ParametersType;
		typedef Superclass1::DerivativeType											DerivativeType; 
		typedef Superclass1::ScalesType													ScalesType; 
		
		/** Typedef's inherited from Elastix.*/
		typedef typename Superclass2::ElastixType						ElastixType;
		typedef typename Superclass2::ElastixPointer				ElastixPointer;
		typedef typename Superclass2::ConfigurationType			ConfigurationType;
		typedef typename Superclass2::ConfigurationPointer	ConfigurationPointer;
		typedef typename Superclass2::RegistrationType			RegistrationType;
		typedef typename Superclass2::RegistrationPointer		RegistrationPointer;
		typedef typename Superclass2::ITKBaseType						ITKBaseType;

		/** Extra typedefs */
		typedef MoreThuenteLineSearchOptimizer							LineOptimizerType;
		typedef LineOptimizerType::Pointer									LineOptimizerPointer;
		typedef ReceptorMemberCommand<Self>									EventPassThroughType;
		typedef typename EventPassThroughType::Pointer			EventPassThroughPointer;
				
		/** Check if any scales are set, and set the UseScales flag on or off; 
		 * after that call the superclass' implementation */
		virtual void StartOptimization(void);

		/** Methods to set parameters and print output at different stages
		 * in the registration process.*/
		virtual void BeforeRegistration(void);
		virtual void BeforeEachResolution(void);
		virtual void AfterEachResolution(void);
		virtual void AfterEachIteration(void);
		virtual void AfterRegistration(void);

		itkGetConstMacro(StartLineSearch, bool);
			
   			
	protected:

		QuasiNewtonLBFGS();
		virtual ~QuasiNewtonLBFGS() {};

		LineOptimizerPointer					m_LineOptimizer;

		/** Convert the line search stop condition to a string */
		virtual std::string GetLineSearchStopCondition(void) const;

		/** Generate a string, representing the phase of optimisation 
		 * (line search, main) */
		virtual std::string DeterminePhase(void) const;

    /** Reimplement the superclass. Calls the superclass' implementation
     * and checks if the MoreThuente line search routine has stopped with
     * Wolfe conditions satisfied. */
    virtual bool TestConvergence( bool firstLineSearchDone );

    /** Call the superclass' implementation. If an ExceptionObject is caught,
     * because the line search optimizer tried a too big step, the exception
     * is printed, but ignored further. The optimizer stops, but elastix
     * just goes on to the next resolution. */
    virtual void LineSearch(
      const ParametersType searchDir,
      double & step,
      ParametersType & x,
      MeasureType & f,
      DerivativeType & g );

	private:

		QuasiNewtonLBFGS( const Self& );	// purposely not implemented
		void operator=( const Self& );							// purposely not implemented

		void InvokeIterationEvent(const EventObject & event);

		EventPassThroughPointer			m_EventPasser;
		double											m_SearchDirectionMagnitude;
		bool												m_StartLineSearch;
		bool												m_GenerateLineSearchIterations;
		bool												m_StopIfWolfeNotSatisfied;
    bool                        m_WolfeIsStopCondition;
			
	}; // end class QuasiNewtonLBFGS
	

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxQuasiNewtonLBFGS.hxx"
#endif

#endif // end #ifndef __elxQuasiNewtonLBFGS_h

