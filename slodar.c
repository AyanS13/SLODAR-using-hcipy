/*
 * SLODAR theoretical impulse response functions
 * Tim Butterley, Richard Wilson, 2008-2022
 *
 */


#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_fit.h>

#include <Python.h>
#include <numpy/arrayobject.h>




/* Kolmogorov structure function
   (r, r_0) */
double Kolstrucfunc(double x, double r0)
{
	return 6.88*pow(x/r0,5./3.);
}



/* Boreman Dainty structure function (needs multiplying by gamma_beta elsewhere) 
   (r, rho_0, beta) */
double BDstrucfunc(double x, double rho, double beta)
{
	return pow(x/rho,beta-2.);
}



/* function to evaluate gamma_beta (Rao et al. 2000) */
double gammabeta(double beta)
{
	double g0,g2,g4,gz;
	
	g0=gsl_sf_gamma(beta/2.);
	g2=gsl_sf_gamma((beta+2.)/2.);
	g4=gsl_sf_gamma((beta+4.)/2.);
	gz=gsl_sf_gamma(beta+1.);
	
	return pow(2.,beta-1.)*g2*g2*g4/(g0*gz);
}



/* Von Karman structure function (Jenkins 1998) 
   (r, r0, mu) */
double VKstrucfunc(double x, double r0, double mu)
{
	double A,B,C,K,brack;

	if (x < 1e-6)
		return 0.;
	else {
		A=2.*pow(M_PI,5./6.)/gsl_sf_gamma(5./6.);
		B=pow(x/mu,5./6.);
		K=gsl_sf_bessel_Knu(5./6.,2.*M_PI*x/mu);
		brack=1.-A*B*K;
		C=pow(mu/r0,5./3.);
		return 0.17253*C*brack;
	}
}



/* Calculate subaperture tilt covariances for Kolmogorov power spectrum
   (nsubx, nsamp, d, theta [deg]) */

static PyObject *slodar_slopecovKol2(PyObject *dummy, PyObject *args)
{
	long     nsubx, nsamp;                /* inputs */
	double   d, theta;
	
	PyArrayObject	*pycov;               /* output */
	npy_intp		pycovDims[3] = {0,0,0};
	double			*pycovData;
	
	//double	d,r0,lamda;
	int      delta, i, j, ia, ja, ib, jb, n2, n3, n4;
	double   th_rad, scaling, x, y, r, dbl_intgrl, phiphi, xtiltcov, ytiltcov;
	double   *tilt, *rxy, *ra_intgrl, *rb_intgrl, *D_phi;
	
	if (!PyArg_ParseTuple(args, "lldd", &nsubx, &nsamp, &d, &theta)) {  //hard-wired for d/r0 = 1, lamda=500nm
		return NULL;
	}
	
	pycovDims[0] = (npy_intp) (nsubx+1)*2;
	pycovDims[1] = (npy_intp) nsubx*2-1;
	pycovDims[2] = (npy_intp) nsubx*2-1;
	pycov = (PyArrayObject *)PyArray_SimpleNew(3, pycovDims, NPY_DOUBLE);    /* create output array */
    pycovData = (double *)PyArray_DATA(pycov);

	//pycov = (PyArrayObject *)PyArray_FromDims(3, dims, NPY_DOUBLE);    /* create output array */
	
	/* allocate arrays needed for calculation */
	n2=nsamp*nsamp;
	n3=n2*nsamp;
	n4=n3*nsamp;
	tilt=calloc(nsamp, sizeof(double));
	rxy=calloc(nsamp, sizeof(double));
	ra_intgrl=calloc(n2, sizeof(double));
	rb_intgrl=calloc(n2, sizeof(double));
	D_phi=calloc(n4, sizeof(double));
	
	scaling = 206265.*206265.*3.*pow(500.e-9/(M_PI*d),2.);
	th_rad = M_PI*theta/180.;

	for(i = 0; i < nsamp; ++i) {
		rxy[i] = ((double)(i-(nsamp/2)) + 0.5)/((double) nsamp);
		tilt[i] = 2.*sqrt(3.)*rxy[i];
	}

	for (delta=0; delta<nsubx+1; ++delta) {
		for (i=0; i<2*nsubx-1; ++i) {
			for (j=0; j<2*nsubx-1; ++j) {
				/* printf("%d %d\n",i,j); */
	
				dbl_intgrl=0.;
				for (ia = 0; ia < nsamp; ++ia) {
					for (ja = 0; ja < nsamp; ++ja) {
						for (ib = 0; ib < nsamp; ++ib) {
							for (jb = 0; jb < nsamp; ++jb) {
								x = ((double) i-nsubx+1) + ((double) delta)*sin(th_rad) - rxy[ia] + rxy[ib];
								y = ((double) j-nsubx+1) + ((double) delta)*cos(th_rad) - rxy[ja] + rxy[jb];
								r = sqrt(x*x + y*y);
								D_phi[n3*ia + n2*ja + nsamp*ib + jb] = Kolstrucfunc(r,1.);  //(x,r0/d);
								ra_intgrl[nsamp*ib + jb] += D_phi[n3*ia + n2*ja + nsamp*ib + jb];
								rb_intgrl[nsamp*ia + ja] += D_phi[n3*ia + n2*ja + nsamp*ib + jb];
								dbl_intgrl += D_phi[n3*ia + n2*ja + nsamp*ib + jb];
							}
						}
					}
				}
				xtiltcov=0.;
				ytiltcov=0.;
				for (ia = 0; ia < nsamp; ++ia) {
					for (ja = 0; ja < nsamp; ++ja) {
						for (ib = 0; ib < nsamp; ++ib) {
							for (jb = 0; jb < nsamp; ++jb) {
								phiphi = 0.5*((ra_intgrl[nsamp*ib + jb] + rb_intgrl[nsamp*ia + ja])/((double) n2));
								phiphi -= 0.5*D_phi[n3*ia + n2*ja + nsamp*ib + jb];
								phiphi -= 0.5*dbl_intgrl/pow((double) nsamp,4.);
								xtiltcov += phiphi*tilt[ia]*tilt[ib];
								ytiltcov += phiphi*tilt[ja]*tilt[jb];
							}
						}
					}
				}
				
				pycovData[ 2*delta    *(nsubx*2-1)*(nsubx*2-1) + i*(nsubx*2-1) + j]=scaling*xtiltcov/((double) n4);
				pycovData[(2*delta+1) *(nsubx*2-1)*(nsubx*2-1) + i*(nsubx*2-1) + j]=scaling*ytiltcov/((double) n4);

				// *(double *)(pycov->data+2*delta*pycov->strides[0]+i*pycov->strides[1]+j*pycov->strides[2]) = scaling*xtiltcov/((double) n4);
				// *(double *)(pycov->data+(2*delta+1)*pycov->strides[0]+i*pycov->strides[1]+j*pycov->strides[2]) = scaling*ytiltcov/((double) n4);
			}
		}
	}
	
	/* deallocate arrays */
	free(tilt);
	free(rxy);
	free(ra_intgrl);
	free(rb_intgrl);
	free(D_phi);
	
	return PyArray_Return(pycov);
}



/* Calculate subaperture tilt covariances for Von Karman power spectrum
	(nsubx, nsamp, d, L_0, theta [deg]) */

static PyObject *slodar_slopecovVK2(PyObject *dummy, PyObject *args)
{
	long     nsubx, nsamp;                /* inputs */
	double   d, L0, theta;
	
	PyArrayObject   *pycov;               /* output */
	npy_intp		pycovDims[3] = {0,0,0};
	double			*pycovData;
	
	//double	d,r0,lamda;
	int      delta, i, j, ia, ja, ib, jb, n2, n3, n4;
	double   th_rad, scaling, x, y, r, dbl_intgrl, phiphi, xtiltcov, ytiltcov;
	double   *tilt, *rxy, *ra_intgrl, *rb_intgrl, *D_phi;
	
	if (!PyArg_ParseTuple(args, "llddd", &nsubx, &nsamp, &d, &L0, &theta)) {  //hard-wired for d/r0 = 1, lamda=500nm
		return NULL;
	}
	
	pycovDims[0] = (npy_intp) (nsubx+1)*2;
	pycovDims[1] = (npy_intp) nsubx*2-1;
	pycovDims[2] = (npy_intp) nsubx*2-1;
	pycov = (PyArrayObject *)PyArray_SimpleNew(3, pycovDims, NPY_DOUBLE);    /* create output array */
    pycovData = (double *)PyArray_DATA(pycov);

	//pycov = (PyArrayObject *)PyArray_FromDims(3, dims, NPY_DOUBLE);    /* create output array */
	
	/* allocate arrays needed for calculation */
	n2=nsamp*nsamp;
	n3=n2*nsamp;
	n4=n3*nsamp;
	tilt=calloc(nsamp, sizeof(double));
	rxy=calloc(nsamp, sizeof(double));
	ra_intgrl=calloc(n2, sizeof(double));
	rb_intgrl=calloc(n2, sizeof(double));
	D_phi=calloc(n4, sizeof(double));
	
	scaling = 206265.*206265.*3.*pow(500.e-9/(M_PI*d),2.);
	th_rad = M_PI*theta/180.;

	for(i = 0; i < nsamp; ++i) {
		rxy[i] = ((double)(i-(nsamp/2)) + 0.5)/((double) nsamp);
		tilt[i] = 2.*sqrt(3.)*rxy[i];
	}

	for (delta=0; delta<nsubx+1; ++delta) {
		for (i=0; i<2*nsubx-1; ++i) {
			for (j=0; j<2*nsubx-1; ++j) {
				/* printf("%d %d\n",i,j); */
	
				dbl_intgrl=0.;
				for (ia = 0; ia < nsamp; ++ia) {
					for (ja = 0; ja < nsamp; ++ja) {
						for (ib = 0; ib < nsamp; ++ib) {
							for (jb = 0; jb < nsamp; ++jb) {
								x = ((double) i-nsubx+1) + ((double) delta)*sin(th_rad) - rxy[ia] + rxy[ib];
								y = ((double) j-nsubx+1) + ((double) delta)*cos(th_rad) - rxy[ja] + rxy[jb];
								r = sqrt(x*x + y*y);
								D_phi[n3*ia + n2*ja + nsamp*ib + jb] = VKstrucfunc(r,1.,L0/d);
								ra_intgrl[nsamp*ib + jb] += D_phi[n3*ia + n2*ja + nsamp*ib + jb];
								rb_intgrl[nsamp*ia + ja] += D_phi[n3*ia + n2*ja + nsamp*ib + jb];
								dbl_intgrl += D_phi[n3*ia + n2*ja + nsamp*ib + jb];
							}
						}
					}
				}
				xtiltcov=0.;
				ytiltcov=0.;
				for (ia = 0; ia < nsamp; ++ia) {
					for (ja = 0; ja < nsamp; ++ja) {
						for (ib = 0; ib < nsamp; ++ib) {
							for (jb = 0; jb < nsamp; ++jb) {
								phiphi = 0.5*((ra_intgrl[nsamp*ib + jb] + rb_intgrl[nsamp*ia + ja])/((double) n2));
								phiphi -= 0.5*D_phi[n3*ia + n2*ja + nsamp*ib + jb];
								phiphi -= 0.5*dbl_intgrl/pow((double) nsamp,4.);
								xtiltcov += phiphi*tilt[ia]*tilt[ib];
								ytiltcov += phiphi*tilt[ja]*tilt[jb];
							}
						}
					}
				}

				pycovData[ 2*delta    *(nsubx*2-1)*(nsubx*2-1) + i*(nsubx*2-1) + j]=scaling*xtiltcov/((double) n4);
				pycovData[(2*delta+1) *(nsubx*2-1)*(nsubx*2-1) + i*(nsubx*2-1) + j]=scaling*ytiltcov/((double) n4);

				// *(double *)(pycov->data+2*delta*pycov->strides[0]+i*pycov->strides[1]+j*pycov->strides[2]) = scaling*xtiltcov/((double) n4);
				// *(double *)(pycov->data+(2*delta+1)*pycov->strides[0]+i*pycov->strides[1]+j*pycov->strides[2]) = scaling*ytiltcov/((double) n4);
			}
		}
	}
	
	/* deallocate arrays */
	free(tilt);
	free(rxy);
	free(ra_intgrl);
	free(rb_intgrl);
	free(D_phi);
	
	return PyArray_Return(pycov);
}



/* Calculate subaperture tilt covariances for Boreman-Dainty power spectrum
   (nsubx, nsamp, d, beta, theta [deg]) */

static PyObject *slodar_slopecovBD2(PyObject *dummy, PyObject *args)
{
	long     nsubx, nsamp;                /* inputs */
	double   d, beta, theta;
	
	PyArrayObject   *pycov;               /* output */
	npy_intp		pycovDims[3] = {0,0,0};
	double			*pycovData;
	
	//double	d,r0,lamda;
	int      delta, i, j, ia, ja, ib, jb, n2, n3, n4;
	double   th_rad, scaling, gambet, x, y, r, dbl_intgrl, phiphi, xtiltcov, ytiltcov;
	double   *tilt, *rxy, *ra_intgrl, *rb_intgrl, *D_phi;
	
	if (!PyArg_ParseTuple(args, "llddd", &nsubx, &nsamp, &d, &beta, &theta)) {  //hard-wired for d/rho0 = 1, lamda=500nm
		return NULL;
	}
	
	pycovDims[0] = (npy_intp) (nsubx+1)*2;
	pycovDims[1] = (npy_intp) nsubx*2-1;
	pycovDims[2] = (npy_intp) nsubx*2-1;
	pycov = (PyArrayObject *)PyArray_SimpleNew(3, pycovDims, NPY_DOUBLE);    /* create output array */
    pycovData = (double *)PyArray_DATA(pycov);

	//pycov = (PyArrayObject *)PyArray_FromDims(3, dims, NPY_DOUBLE);    /* create output array */
	
	/* allocate arrays needed for calculation */
	n2=nsamp*nsamp;
	n3=n2*nsamp;
	n4=n3*nsamp;
	tilt=calloc(nsamp, sizeof(double));
	rxy=calloc(nsamp, sizeof(double));
	ra_intgrl=calloc(n2, sizeof(double));
	rb_intgrl=calloc(n2, sizeof(double));
	D_phi=calloc(n4, sizeof(double));
	
	scaling = 206265.*206265.*3.*pow(500.e-9/(M_PI*d),2.);
	th_rad = M_PI*theta/180.;

	for(i = 0; i < nsamp; ++i) {
		rxy[i] = ((double)(i-(nsamp/2)) + 0.5)/((double) nsamp);
		tilt[i] = 2.*sqrt(3.)*rxy[i];
	}

	gambet=gammabeta(beta);
	for (delta=0; delta<nsubx+1; ++delta) {
		for (i=0; i<2*nsubx-1; ++i) {
			for (j=0; j<2*nsubx-1; ++j) {
				/* printf("%d %d\n",i,j); */
	
				dbl_intgrl=0.;
				for (ia = 0; ia < nsamp; ++ia) {
					for (ja = 0; ja < nsamp; ++ja) {
						for (ib = 0; ib < nsamp; ++ib) {
							for (jb = 0; jb < nsamp; ++jb) {
								x = ((double) i-nsubx+1) + ((double) delta)*sin(th_rad) - rxy[ia] + rxy[ib];
								y = ((double) j-nsubx+1) + ((double) delta)*cos(th_rad) - rxy[ja] + rxy[jb];
								r = sqrt(x*x + y*y);
								D_phi[n3*ia + n2*ja + nsamp*ib + jb] = gambet*BDstrucfunc(r,1.,beta);
								ra_intgrl[nsamp*ib + jb] += D_phi[n3*ia + n2*ja + nsamp*ib + jb];
								rb_intgrl[nsamp*ia + ja] += D_phi[n3*ia + n2*ja + nsamp*ib + jb];
								dbl_intgrl += D_phi[n3*ia + n2*ja + nsamp*ib + jb];
							}
						}
					}
				}
				xtiltcov=0.;
				ytiltcov=0.;
				for (ia = 0; ia < nsamp; ++ia) {
					for (ja = 0; ja < nsamp; ++ja) {
						for (ib = 0; ib < nsamp; ++ib) {
							for (jb = 0; jb < nsamp; ++jb) {
								phiphi = 0.5*((ra_intgrl[nsamp*ib + jb] + rb_intgrl[nsamp*ia + ja])/((double) n2));
								phiphi -= 0.5*D_phi[n3*ia + n2*ja + nsamp*ib + jb];
								phiphi -= 0.5*dbl_intgrl/pow((double) nsamp,4.);
								xtiltcov += phiphi*tilt[ia]*tilt[ib];
								ytiltcov += phiphi*tilt[ja]*tilt[jb];
							}
						}
					}
				}

				pycovData[ 2*delta    *(nsubx*2-1)*(nsubx*2-1) + i*(nsubx*2-1) + j]=scaling*xtiltcov/((double) n4);
				pycovData[(2*delta+1) *(nsubx*2-1)*(nsubx*2-1) + i*(nsubx*2-1) + j]=scaling*ytiltcov/((double) n4);

				// *(double *)(pycov->data+2*delta*pycov->strides[0]+i*pycov->strides[1]+j*pycov->strides[2]) = scaling*xtiltcov/((double) n4);
				// *(double *)(pycov->data+(2*delta+1)*pycov->strides[0]+i*pycov->strides[1]+j*pycov->strides[2]) = scaling*ytiltcov/((double) n4);
			}
		}
	}
	
	/* deallocate arrays */
	free(tilt);
	free(rxy);
	free(ra_intgrl);
	free(rb_intgrl);
	free(D_phi);
	
	return PyArray_Return(pycov);
}




/* Calculate tilt-subtracted 2D reference functions for given pupil geometry and arbitrary separation angle
   [Butterley, Wilson & Sarazin, 2006, MNRAS 369, 835-845]
   (cov, pupil, phi) */

static PyObject *slodar_refFuncs2D(PyObject *dummy, PyObject *args)
{
	PyArrayObject   *cov,*pupil;               /* inputs */
	npy_intp		*covDims,*pupilDims;
	double			*covData;
	int				*pupilData;

	PyArrayObject   *psfs;               /* output */
	npy_intp		psfsDims[3] = {0,0,0};
	double			*psfsData;

	int nsubx,nn,nsubtot,nsirfs;  //ncov,
	int	i,j,i1,j1,i2,j2,isub1,isub2,di,dj,delta;
	double mcovx,mcovy;
	double *pcov,*pcov2,*pcovtt,*pcovtt2;
	int *psfcount;


	if (!PyArg_ParseTuple (args, "O!O!", &PyArray_Type, &cov, &PyArray_Type, &pupil)){
	  return NULL;
	}
	if (PyArray_TYPE(cov) != NPY_DOUBLE || PyArray_TYPE(pupil) != NPY_INT32) {
	    printf("slodar.slopecovPSF2D: Input array(s) with incorrect data type\n");
	    return NULL;
	}
	if (!PyArray_ISCONTIGUOUS(cov) || !PyArray_ISCONTIGUOUS(pupil)) {
	    printf("slodar.slopecovPSF2D: Input array(s) non-contiguous\n");
	    return NULL;
	}
	
	covDims=PyArray_DIMS(cov);
	pupilDims=PyArray_DIMS(pupil);
	covData=PyArray_DATA(cov);
	pupilData=PyArray_DATA(pupil);


	nsubx=(int)pupilDims[0]; //pupil->dimensions[0];
	//ncov=cov->dimensions[0];
	nn=2*nsubx-1;
	nsirfs=(int)covDims[0]/2;
	
	nsubtot=0;
	for (j=0;j<nsubx;++j) {
		for (i=0;i<nsubx;++i) {
			//if (*(int *)(pupil->data+j*pupil->strides[0]+i*pupil->strides[1]) > 0) {
			if (pupilData[j*nsubx+i] > 0) {
				++nsubtot;
			}
		}
	}
	//printf("%d\n",nsubtot);

	pcov=calloc(nsubtot*nsubtot*2, sizeof(double));		//pupil-projected covariance
	pcov2=calloc(nsubtot*nsubtot*2, sizeof(double));	//tip/tilt subtracted pupil-projected covariance
	pcovtt=calloc(nsubtot*2, sizeof(double));
	pcovtt2=calloc(nsubtot*2, sizeof(double));
	psfcount=calloc(nn*nn, sizeof(int));
	
	psfsDims[0]=(npy_intp)(2*nsirfs);  // 2*nsubx;
	psfsDims[1]=(npy_intp)nn;
	psfsDims[2]=(npy_intp)nn;
	psfs = (PyArrayObject *)PyArray_SimpleNew(3, psfsDims, NPY_DOUBLE);    /* create output array */
	psfsData=PyArray_DATA(psfs);
	for (i=0; i<2*nsirfs*nn*nn; ++i) {
		psfsData[i]=0.;
	}
	
	for (delta=0; delta<nsirfs; ++delta) {
		isub1=0;
		for (j1=0; j1<nsubx; ++j1) {
			for (i1=0; i1<nsubx; ++i1) {
				//if (*(int *)(pupil->data+j1*pupil->strides[0]+i1*pupil->strides[1]) > 0) {
				if (pupilData[j1*nsubx+i1] > 0) {
					isub2=0;
					for (j2=0; j2<nsubx; ++j2) {
						for (i2=0; i2<nsubx; ++i2) {
							//if (*(int *)(pupil->data+j2*pupil->strides[0]+i2*pupil->strides[1]) > 0) {
							if (pupilData[j2*nsubx+i2] > 0) {
								dj=j2-j1+nsubx-1;
								di=i2-i1+nsubx-1;
								pcov[isub1*nsubtot*2+isub2*2]=covData[2*delta*nn*nn + di*nn + dj];
								pcov[isub1*nsubtot*2+isub2*2+1]=covData[(2*delta+1)*nn*nn + di*nn + dj];
								//pcov[isub1*nsubtot*2+isub2*2]=*(double *)(cov->data+2*delta*cov->strides[0]+di*cov->strides[1]+dj*cov->strides[2]);
								//pcov[isub1*nsubtot*2+isub2*2+1]=*(double *)(cov->data+(2*delta+1)*cov->strides[0]+di*cov->strides[1]+dj*cov->strides[2]);
								isub2++;
							}
						}
					}
					isub1++;
				}
			}
		}
		mcovx=0.;
		mcovy=0.;
		for (i=0; i<nsubtot*2; ++i) {
			pcovtt[i]=(double) 0.;
			pcovtt2[i]=(double) 0.;
		}		
		for (j=0; j<nsubtot; ++j) {
			for (i=0; i<nsubtot; ++i) {
				pcovtt[j*2] += pcov[j*nsubtot*2+i*2];
				pcovtt[j*2+1] += pcov[j*nsubtot*2+i*2+1];
				pcovtt2[j*2] += pcov[i*nsubtot*2+j*2];
				pcovtt2[j*2+1] += pcov[i*nsubtot*2+j*2+1];
				mcovx += pcov[j*nsubtot*2+i*2];
				mcovy += pcov[j*nsubtot*2+i*2+1];
			}
		}
		for (i=0; i<nsubtot*2; ++i) {
			pcovtt[i] /= (double) nsubtot;
			pcovtt2[i] /= (double) nsubtot;
			//printf("delta = %d, i = %d, covtt = %5.3le %5.3le\n",delta,i,pcovtt[i],pcovtt2[i]);
		}		
		mcovx /= (double) (nsubtot*nsubtot);
		mcovy /= (double) (nsubtot*nsubtot);

		//printf("delta = %d, mcov = %5.3le %5.3le\n",delta,mcovx,mcovy);

		for (j=0; j<nsubtot; ++j) {
			for (i=0; i<nsubtot; ++i) {
				pcov2[j*nsubtot*2+i*2] = pcov[j*nsubtot*2+i*2] - pcovtt[j*2] - pcovtt2[i*2] + mcovx;
				pcov2[j*nsubtot*2+i*2+1] = pcov[j*nsubtot*2+i*2+1] - pcovtt[j*2+1] - pcovtt2[i*2+1] + mcovy;
			}
		}

		for (j=0; j<nn; ++j) {
			for (i=0; i<nn; ++i) {
				psfcount[j*nn+i]=0;
			}
		}
		isub1=0;
		for (j1=0;j1<nsubx;++j1) {
			for (i1=0;i1<nsubx;++i1) {
				//if (*(int *)(pupil->data+j1*pupil->strides[0]+i1*pupil->strides[1]) > 0) {
				if (pupilData[j1*nsubx+i1] > 0) {
					isub2=0;
					for (j2=0;j2<nsubx;++j2) {
						for (i2=0;i2<nsubx;++i2) {
							//if (*(int *)(pupil->data+j2*pupil->strides[0]+i2*pupil->strides[1]) > 0) {
							if (pupilData[j2*nsubx+i2] > 0) {
								di=(i2-i1)+(nsubx-1);
								dj=(j2-j1)+(nsubx-1);
								// X AND Y SWAPPED BELOW!
								psfsData[2*delta*nn*nn + dj*nn + di] += pcov2[isub1*nsubtot*2+isub2*2];
								psfsData[(2*delta+1)*nn*nn + dj*nn + di] += pcov2[isub1*nsubtot*2+isub2*2+1];
								// *(double *)(psfs->data+2*delta*psfs->strides[0]+dj*psfs->strides[1]+di*psfs->strides[2]) += pcov2[isub1*nsubtot*2+isub2*2];	
								// *(double *)(psfs->data+(2*delta+1)*psfs->strides[0]+dj*psfs->strides[1]+di*psfs->strides[2]) += pcov2[isub1*nsubtot*2+isub2*2+1];

								psfcount[dj*nn+di]+=1;
								//psfcount[dj+nn]+=1;	
								isub2++;
							}
						}
					}
					isub1++;
				}
			}
		}

		for (j=0; j<nn; ++j) {
			for (i=0; i<nn; ++i) {
				//printf("%d ",psfcount[i]);
				if (psfcount[j*nn+i]>0) {
					psfsData[2*delta*nn*nn + j*nn + i] /= (double) psfcount[j*nn+i];
					psfsData[(2*delta+1)*nn*nn + j*nn + i] /= (double) psfcount[j*nn+i];
					
					// *(double *)(psfs->data+2*delta*psfs->strides[0]+j*psfs->strides[1]+i*psfs->strides[2]) /= (double) psfcount[j*nn+i];
					// *(double *)(psfs->data+(2*delta+1)*psfs->strides[0]+j*psfs->strides[1]+i*psfs->strides[2]) /= (double) psfcount[j*nn+i];
				}
			}
		}
		//printf("\n");

	}

	free(pcov);
	free(pcov2);
	free(pcovtt);
	free(pcovtt2);
	free(psfcount);
	return PyArray_Return(psfs);
}






// Method definition object for this extension, these arguments mean:
// ml_name: The name of the method
// ml_meth: Function pointer to the method implementation
// ml_flags: Flags indicating special features of this method, such as
//          accepting arguments, accepting keyword arguments, being a
//          class method, or being a static method of a class.
// ml_doc:  Contents of this method's docstring
static PyMethodDef slodar_methods[] = { 
    {   
        "slopecovKol", slodar_slopecovKol2, METH_VARARGS,
        "Calculate slope covariances for Kolmogorov turbulence"
    },  
    {   
        "slopecovVK", slodar_slopecovVK2, METH_VARARGS,
        "Calculate slope covariances for Von Karman turbulence"
    },  
    {   
        "slopecovBD", slodar_slopecovBD2, METH_VARARGS,
        "Calculate slope covariances for Boreman-Dainty turbulence"
    },  
    {   
        "refFuncs", slodar_refFuncs2D, METH_VARARGS,
        "Generate 2D reference functions with tip/tilt subtracted"
    },  
    {NULL, NULL, 0, NULL}
};


// Module definition
// The arguments of this structure tell Python what to call your extension,
// what it's methods are and where to look for it's method definitions
static struct PyModuleDef slodar_definition = { 
    PyModuleDef_HEAD_INIT,
    "slodar",
    "C extension for Python to generate SLODAR reference functions.",
    -1, 
    slodar_methods
};


// Module initialization
// Python calls this function when importing your extension. It is important
// that this function is named PyInit_[[your_module_name]] exactly, and matches
// the name keyword argument in setup.py's setup() call.
PyMODINIT_FUNC PyInit_slodar(void) {
    Py_Initialize();
    import_array();
    return PyModule_Create(&slodar_definition);
}





