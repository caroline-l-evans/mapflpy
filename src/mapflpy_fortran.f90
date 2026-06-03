
! Version 1.0.0

module mapfl

implicit none

character(256) :: set_me
namelist /test/ set_me

contains

subroutine run (debug_level_, &
                use_analytic_function_, &
                function_params_file_, &
                domain_r_min_, &
                domain_r_max_, &
                bfile_r_, &
                bfile_t_, &
                bfile_p_, &
                cubic_, &
                ds_variable_, &
                ds_min_, &
                ds_max_, &
                ds_limit_by_local_mesh_, &
                ds_local_mesh_factor_, &
                ds_over_rc_, &
                ds_lmax_, &
                set_ds_automatically_, &
                dsmult_, trace_fwd_, trace_bwd_, &
                rffile_, tffile_, pffile_, effile_, kffile_, qffile_, lffile_, &
                rbfile_, tbfile_, pbfile_, ebfile_, kbfile_, qbfile_, lbfile_, &
                new_r_mesh_, mesh_file_r_, nrss_, r0_,r1_, new_t_mesh_, mesh_file_t_, &
                ntss_, t0_,t1_, new_p_mesh_, mesh_file_p_, npss_, p0_,p1_, trace_3d_, &
                volume3d_output_file_r_, volume3d_output_file_t_, volume3d_output_file_p_, &
                trace_slice_, slice_coords_are_xyz_, &
                trace_slice_direction_is_along_b_, compute_q_on_slice_, &
                q_increment_h_, &
                slice_input_file_r_, slice_input_file_t_, slice_input_file_p_, &
                trace_from_slice_forward_, &
                slice_output_file_forward_r_, slice_output_file_forward_t_, slice_output_file_forward_p_, &
                trace_from_slice_backward_, &
                slice_output_file_backward_r_, slice_output_file_backward_t_, slice_output_file_backward_p_, &
                slice_q_output_file_, &
                slice_length_output_file_, compute_ch_map_, ch_map_r_, &
                ch_map_output_file_, compute_ch_map_3d_, ch_map_3d_output_file_, &
                write_traces_to_hdf_, write_traces_root_, write_traces_as_xyz_, &
                integrate_along_fl_, scalar_input_file_, &
                weight_integral_by_area_, max_along_fl_, &
                verbose_, &
                br, br_p, br_t, br_r, br_np, br_nt, br_nr, &
                bt, bt_p, bt_t, bt_r, bt_np, bt_nt, bt_nr, &
                bp, bp_p, bp_t, bp_r, bp_np, bp_nt, bp_nr)

      ! compute_dips_map_3d_
      ! dips_map_3d_output_file_, ns_dips_, &
      ! slogqffile_, slogqbfile_

      use ident
      use params
      use types
      use files
      use mesh
      use field
      use vars
      use field_line_params
      use step_size_stats
      use debug
! c
! c-----------------------------------------------------------------------
! c
      implicit none
! c
! c-----------------------------------------------------------------------
! c
      integer :: ierr,i
      real(r_typ) :: ch_map_r=1.0_r_typ
      logical :: trace_fwd=.false.,trace_bwd=.false.
      logical :: compute_q_on_slice=.false.
      logical :: compute_ch_map=.false.
      logical :: compute_ch_map_3d=.false.
      logical :: compute_dips_map_3d=.false.
      character(256) :: errline=' '
! 
!  input parameters should end in underscore
      integer :: debug_level_
      logical :: use_analytic_function_
      character(512) :: function_params_file_
      real(r_typ) :: domain_r_min_
      real(r_typ) :: domain_r_max_
      character(512) :: bfile_r_, bfile_t_, bfile_p_
      logical :: cubic_
      logical :: ds_variable_, ds_limit_by_local_mesh_
      real(r_typ) :: ds_min_, ds_max_, ds_local_mesh_factor_, ds_lmax_, ds_over_rc_
      logical :: set_ds_automatically_
      real(r_typ) :: dsmult_
      logical :: trace_fwd_, trace_bwd_
      character(512) :: rffile_, tffile_, pffile_
      character(512) :: effile_, kffile_, qffile_, lffile_
      character(512) :: rbfile_, tbfile_, pbfile_, ebfile_
      character(512) :: kbfile_, qbfile_, lbfile_
      logical :: new_r_mesh_, new_t_mesh_, new_p_mesh_
      character(512) :: mesh_file_r_, mesh_file_p_, mesh_file_t_
      integer :: nrss_, ntss_, npss_
      real(r_typ) :: r0_, r1_, t0_, t1_, p0_, p1_
      logical :: trace_3d_
      character(512) :: volume3d_output_file_r_, volume3d_output_file_t_, volume3d_output_file_p_
      logical :: trace_slice_
      logical :: slice_coords_are_xyz_
      logical :: trace_slice_direction_is_along_b_
      logical :: compute_q_on_slice_
      real(r_typ) :: q_increment_h_
      character(512) :: slice_input_file_r_, slice_input_file_t_, slice_input_file_p_
      logical :: trace_from_slice_forward_
      character(512) :: slice_output_file_forward_r_, slice_output_file_forward_t_, slice_output_file_forward_p_
      logical :: trace_from_slice_backward_
      character(512) :: slice_output_file_backward_r_, slice_output_file_backward_t_, slice_output_file_backward_p_
      character(512) :: slice_q_output_file_
      character(512) :: slice_length_output_file_
      logical :: compute_ch_map_
      real(r_typ) :: ch_map_r_
      character(512) :: ch_map_output_file_
      character(512) :: ch_map_3d_output_file_
      logical :: compute_ch_map_3d_
      ! logical :: compute_dips_map_3d_
      logical :: write_traces_to_hdf_
      character(64) :: write_traces_root_
      logical :: write_traces_as_xyz_
      character(512) :: dips_map_3d_output_file_
      integer :: ns_dips_
      character(512) :: slogqffile_
      character(512) :: slogqbfile_
      logical :: integrate_along_fl_
      character(512) :: scalar_input_file_
      logical :: weight_integral_by_area_
      logical :: max_along_fl_

      logical :: verbose_
      character(100) :: confirm

      integer, intent(in) :: br_nr, br_nt, br_np
      integer, intent(in) :: bt_nr, bt_nt, bt_np
      integer, intent(in) :: bp_nr, bp_nt, bp_np

      real(r_typ), intent(in), target :: br_p(br_np), br_t(br_nt), br_r(br_nr)
      real(r_typ), intent(in), target :: bt_p(bt_np), bt_t(bt_nt), bt_r(bt_nr)
      real(r_typ), intent(in), target :: bp_p(bp_np), bp_t(bp_nt), bp_r(bp_nr)
      
      ! according to Cooper, MAS has resolution like so
      real(r_typ), intent(inout), target :: br(br_nr, br_nt, br_np)
      real(r_typ), intent(inout), target :: bt(bt_nr, bt_nt, bt_np)
      real(r_typ), intent(inout), target :: bp(bp_nr, bp_nt, bp_np)

      namelist /datum/ &
        debug_level, use_analytic_function, function_params_file, &
        domain_r_min, domain_r_max, bfile, &
        cubic, ds, set_ds_automatically, &
        dsmult, trace_fwd, trace_bwd, &
        rffile, tffile, pffile, effile, kffile, qffile, lffile, &
        rbfile, tbfile, pbfile, ebfile, kbfile, qbfile, lbfile, &
        new_r_mesh, mesh_file_r, nrss, r0,r1, new_t_mesh, mesh_file_t, &
        ntss, t0, t1, new_p_mesh, mesh_file_p, npss, p0, p1, trace_3d, &
        volume3d_output_file, trace_slice, slice_coords_are_xyz, &
        trace_slice_direction_is_along_b, compute_q_on_slice, &
        q_increment_h, slice_input_file, trace_from_slice_forward, &
        slice_output_file_forward, trace_from_slice_backward, &
        slice_output_file_backward, slice_q_output_file, &
        slice_length_output_file, compute_ch_map, ch_map_r, &
        ch_map_output_file, compute_ch_map_3d, ch_map_3d_output_file, &
        write_traces_to_hdf, write_traces_root, write_traces_as_xyz, &
        compute_dips_map_3d, dips_map_3d_output_file, ns_dips, &
        slogqffile, slogqbfile, integrate_along_fl, scalar_input_file

      
      debug_level = debug_level_
      use_analytic_function = use_analytic_function_
      function_params_file = function_params_file_
      domain_r_min = domain_r_min_
      domain_r_max = domain_r_max_
      bfile%r = bfile_r_
      bfile%t = bfile_t_
      bfile%p = bfile_p_
      cubic = cubic_

      ds%variable = ds_variable_
      ds%min = ds_min_
      ds%max = ds_max_
      ds%limit_by_local_mesh = ds_limit_by_local_mesh_
      ds%local_mesh_factor = ds_local_mesh_factor_
      ds%lmax = ds_lmax_
      ds%over_rc = ds_over_rc_

      set_ds_automatically = set_ds_automatically_

      dsmult = dsmult_
      trace_fwd = trace_fwd_
      trace_bwd = trace_bwd_

      rffile = rffile_
      tffile = tffile_
      pffile = pffile_
      effile = effile_
      kffile = kffile_
      qffile = qffile_
      lffile = lffile_

      rbfile = rbfile_
      tbfile = tbfile_
      pbfile = pbfile_
      ebfile = ebfile_
      kbfile = kbfile_
      qbfile = qbfile_
      lbfile = lbfile_

      new_r_mesh = new_r_mesh_
      mesh_file_r = mesh_file_r_
      nrss = nrss_
      r0 = r0_
      r1 = r1_
      new_t_mesh = new_t_mesh_
      mesh_file_t = mesh_file_t_

      ntss = ntss_
      t0 = t0_
      t1 = t1_
      new_p_mesh = new_p_mesh_
      mesh_file_p = mesh_file_p_
      npss = npss_
      p0 = p0_
      p1 = p1_
      trace_3d = trace_3d_

      volume3d_output_file%r = volume3d_output_file_r_
      volume3d_output_file%t = volume3d_output_file_t_
      volume3d_output_file%p = volume3d_output_file_p_
      trace_slice = trace_slice_
      slice_coords_are_xyz = slice_coords_are_xyz_

      trace_slice_direction_is_along_b = trace_slice_direction_is_along_b_
      compute_q_on_slice = compute_q_on_slice_

      q_increment_h = q_increment_h_
      slice_input_file%r = slice_input_file_r_
      slice_input_file%t = slice_input_file_t_
      slice_input_file%p = slice_input_file_p_
      trace_from_slice_forward = trace_from_slice_forward_

      slice_output_file_forward%r = slice_output_file_forward_r_
      slice_output_file_forward%t = slice_output_file_forward_t_
      slice_output_file_forward%p = slice_output_file_forward_p_
      trace_from_slice_backward = trace_from_slice_backward_

      slice_output_file_backward%r = slice_output_file_backward_r_
      slice_output_file_backward%t = slice_output_file_backward_t_
      slice_output_file_backward%p = slice_output_file_backward_p_
      slice_q_output_file = slice_q_output_file_

      slice_length_output_file = slice_length_output_file_
      compute_ch_map = compute_ch_map_
      ch_map_r = ch_map_r_

      ch_map_output_file = ch_map_output_file_
      compute_ch_map_3d = compute_ch_map_3d_
      ch_map_3d_output_file = ch_map_3d_output_file_

      write_traces_to_hdf = write_traces_to_hdf_
      write_traces_root = write_traces_root_
      write_traces_as_xyz = write_traces_as_xyz_

      !  These parameters not available in all mapfl.in
      ! compute_dips_map_3d = compute_dips_map_3d_
      ! dips_map_3d_output_file = dips_map_3d_output_file_
      ! ns_dips = ns_dips_

      ! slogqffile = slogqffile_
      ! slogqbfile = slogqbfile_
      integrate_along_fl = integrate_along_fl_
      scalar_input_file = scalar_input_file_
      weight_integral_by_area = weight_integral_by_area_
      max_along_fl = max_along_fl_

      verbose = verbose_

      if (verbose.gt.0) then
        write (*,100) debug_level
      end if

      100 FORMAT (I4)


      if (verbose.gt.0) then
        write (*,*) br_np, br_nt, br_nr, br(5,5,5)
        write (*,*) bt_np, bt_nt, bt_nr, bt(5,5,5)
        write (*,*) bp_np, bp_nt, bp_nr, bp(5,5,5)
      end if

! c
! c ****** Set the parameters.
! c
!       call set_parameters

!       call ffopen (1,infile,'r',ierr)

!       if (ierr.ne.0) then
!         write (*,*)
!         write (*,*) '### ERROR in MAPFL:'
!         write (*,*) '### The input file does not exist'//
!      &              ' or cannot be read.'
!         write (*,*) 'File name: ',trim(infile)
!         call exit (1)
!       end if
! ! c
! ! c ****** Read the input file.
! ! c
!       call ffopen (1,trim(infile),'r',ierr)
!       read(1,datum,iostat=ierr)
!       if (ierr.ne.0) then
!         backspace (1)
!         read (1,fmt='(A)') errline
!         write (*,*)
!         write (*,*) '### ERROR reading input file:'
!         write (*,*) '### The following line has a problem:'
!         write (*,*)
!         write (*,*) trim(errline)
!         write (*,*)
!         write (*,*) '###'
!         call exit (1)
!       endif
!       write (*,*)
!       write (*,*) '### Input file contents:'
!       write (*,*)
!       write(*,datum)
!       close (1)

      if (verbose.gt.0) then
        write (*,*)
        write (*,*) '### ',cname,' Version ',cvers,' of ',cdate,'.'
      end if
! c
! c ****** Read the parameters that define the analytic magnetic
! c ****** field function, if requested.
! c
      if (use_analytic_function) then
! c        call read_function_params
        write (*,*)
        write (*,*) '### ERROR in MAPFLPY:'
        write (*,*) '### ANALYTIC FUNCTIONS NOT CURRENTLY IMPLEMENTED IN THIS VERSION'
        call exit(1)
      end if
! c
! c ****** Set the field line integration parameters.
! c
      ds%max_increase_factor=max_increase_factor
      ds%max_decrease_factor=max_decrease_factor
      ds%predictor_min_clip_fraction=predictor_min_clip_fraction
      ds%short_fl_min_points=short_fl_min_points
      ds%short_fl_max_tries=short_fl_max_tries
      ds%short_fl_shrink_factor=short_fl_shrink_factor

      if (verbose.gt.0) then
        write (*,*) 'ds values:', ds
      end if
! c
! c ****** Read the magnetic field.
! c

      if (verbose.gt.0) then
        print*, "use use_analytic_function:", use_analytic_function_
        print*,"about to read magnetic field"
      end if

      if (.not.use_analytic_function) then

        !!!!!! br !!!!!!!!!!
        if (verbose.gt.0) then
          print*, "setting dimensions for r"
        end if

        b%r%ndim = 3
        b%r%dims = (/ br_nr, br_nt, br_np /)
        b%r%scale = .true.
        b%r%hdf32 = .true.

        if (verbose.gt.0) then
          print*, "setting scales"
        end if

        b%r%scales(1)%f => br_r
        b%r%scales(2)%f => br_t
        b%r%scales(3)%f => br_p

        if (verbose.gt.0) then
          print*, "assigning field component br"
        end if
        b%r%f => br

        !!!!!! bp !!!!!!!!!!
        if (verbose.gt.0) then
          print*, "setting dimensions for p"
        end if
        b%p%ndim = 3
        b%p%dims = (/ bp_nr, bp_nt, bp_np  /)
        b%p%scale = .true.
        b%p%hdf32 = .true.

        if (verbose.gt.0) then
          print*, "setting scales"
        end if

        b%p%scales(1)%f => bp_r
        b%p%scales(2)%f => bp_t
        b%p%scales(3)%f => bp_p

        if (verbose.gt.0) then
          print*, "assigning field component bp"
        end if
        b%p%f => bp

        !!!!!! bt !!!!!!!!!!
        if (verbose.gt.0) then
          print*, "setting dimensions for p"
        end if
        b%t%ndim = 3
        b%t%dims = (/ bt_nr, bt_nt, bt_np /)
        b%t%scale = .true.
        b%t%hdf32 = .true.

        if (verbose.gt.0) then
          print*, "setting scales"
        end if

        b%t%scales(1)%f => bt_r
        b%t%scales(2)%f => bt_t
        b%t%scales(3)%f => bt_p

        if (verbose.gt.0) then
          print*, "assigning field component bt"
        end if
        b%t%f => bt


        ! b%r%f(5,5,5) = 2
        ! print*, 'changed br(5,5,5)', b%r%f(5,5,5)

        ! call readb (bfile,b)
        if (verbose.gt.0) then
          print*, "calling set_btype"
        end if
        call set_btype(b)

        if (verbose.gt.0) then
          print*, "creating component inverse tables"
        end if
        call build_inverse_tables (b%r,b%inv(1))
        call build_inverse_tables (b%t,b%inv(2))
        call build_inverse_tables (b%p,b%inv(3))

        if (cubic) then
          b%cubic=.true.
          if (verbose.gt.0) then
            write (*,*)
            write (*,*) 'Computing cubic spline coefficients for Br ...'
          end if
          call compute_spline_3d (b%r%dims(1),b%r%dims(2),b%r%dims(3), &
                                  b%r%scales(1)%f, &
                                  b%r%scales(2)%f, &
                                  b%r%scales(3)%f, &
                                  b%r%f,b%spl%r)
          if (verbose.gt.0) then
            write (*,*)
            write (*,*) 'Computing cubic spline coefficients for Bt ...'
          end if
          call compute_spline_3d (b%t%dims(1),b%t%dims(2),b%t%dims(3), &
                                  b%t%scales(1)%f, &
                                  b%t%scales(2)%f, &
                                  b%t%scales(3)%f, &
                                  b%t%f,b%spl%t)
          if (verbose.gt.0) then
            write (*,*)
            write (*,*) 'Computing cubic spline coefficients for Bp ...'
          end if
          call compute_spline_3d (b%p%dims(1),b%p%dims(2),b%p%dims(3), &
                                 b%p%scales(1)%f, &
                                 b%p%scales(2)%f, &
                                 b%p%scales(3)%f, &
                                 b%p%f,b%spl%p)
        else
          if (verbose.gt.0) then
            print*, "cubic not requested"
          end if
          b%cubic=.false.
        end if

      end if

! This part should not be necessary if we are not writing output to file
! however, if we are then we'll need to allow these
! c
! c ****** Set the trace output format based on input br
! c ****** (for analytic function, sets to hdf)
! c
      i=index(bfile%r,'.h');
      if (bfile%r(i+1:i+2).eq.'h5') then
        fmt='h5'
      endif

! ! c
! ! c ****** Set the radial domain limits to those specified.
! ! c
      b%lim0(1)=max(b%lim0(1),domain_r_min)
      b%lim1(1)=min(b%lim1(1),domain_r_max)

      if (verbose.gt.0) then
        write (*,*)
        write (*,*) '### Domain limits:'
        write (*,*) 'Lower boundary value: ',b%lim0(1)
        write (*,*) 'Upper boundary value: ',b%lim1(1)
      end if
   
! c
! c ****** Make the new r, t, and p meshes.
! c
      call make_new_meshes (b)
! ! c
! ! c ****** Set the default step size.
! ! c
      call set_ds (b)
! ! c
! ! c ****** Set the flag to gather step size statistics.
! ! c
      gather_stats=verbose
! ! c
! ! c ****** Setup the field to integrate along if requested.
! ! c
      if (integrate_along_fl.or.max_along_fl) call set_up_scalar_field
! ! c
! ! c ****** Trace the field lines forward, set dir and/or map if requested.
! ! c
      if (trace_fwd) then
        ds%direction=1
        if (rffile.ne.' '.or. &
            tffile.ne.' '.or. &
            pffile.ne.' '.or. &
            effile.ne.' '.or. &
            kffile.ne.' '.or. &
            qffile.ne.' '.or. &
            lffile.ne.' ') then
          call map_forward
        endif
      endif
! c
! c ****** Trace the field lines backward, set dir and/or map if requested.
! c
      if (trace_bwd) then
        ds%direction=-1
        if (rbfile.ne.' '.or. &
            tbfile.ne.' '.or. &
            pbfile.ne.' '.or. &
            ebfile.ne.' '.or. &
            kbfile.ne.' '.or. &
            qbfile.ne.' '.or. &
            lbfile.ne.' ') then
          call map_backward
        endif
      endif
! c
! c ****** Map the field lines from a 3D rectilinear volume,
! c ****** if requested.
! c
      if (trace_3d) call map_3d
! c
! c ****** Map the field lines from a slice, if requested,
! c ****** or determine Q on the slice, if requested.
! c
      if (trace_slice) then
        call read_slice_coordinates
        if (compute_q_on_slice) then
          call get_q_on_slice
        else
          call map_slice
        end if
        if (verbose.gt.0) then
          print*, "deallocating slice coordinates"
        end if
        call deallocate_slice_coordinates
      end if
! c
! c ****** Compute a coronal hole map, if requested.
! c
      if (compute_ch_map) then
        call get_ch_map (ch_map_r)
      end if
! c
! c ****** Compute a 3D coronal hole map, if requested.
! c
      if (compute_ch_map_3d) then
        call get_ch_map_3d
      end if
! c
! c ****** Compute a 3D dips map, if requested.
! c
      if (compute_dips_map_3d) then
        call get_dips_map_3d
      end if

      if (verbose.gt.0) then
        stat_ds_avg=0.
        if (stat_n.ne.0) stat_ds_avg=stat_ds_sum/stat_n
        write (*,*)
        write (*,*) '### Field line integration step size statistics:'
        write (*,*) 'Number of field line segments = ',stat_n
        write (*,*) 'Minimum step size used = ',stat_ds_min
        write (*,*) 'Maximum step size used = ',stat_ds_max
        write (*,*) 'Average step size used = ',stat_ds_avg
      end if

    if (verbose.gt.0) then
      print*, "finished!"
    end if

    end subroutine run

subroutine init (cubic_vec_field, &
                 var_dstep, &
                 dstep, &
                 auto_minmax_dstep, &
                 min_dstep, &
                 max_dstep, &
                 dstep_mult, &
                 limit_by_local_mesh, &
                 local_mesh_factor, &
                 max_length, &
                 direction_along_vec_field, &
                 trace_from_slice_forward, &
                 trace_from_slice_backward)

  use types
  use ident
  ! use visual

  implicit none

  logical :: cubic_vec_field
  logical :: var_dstep
  real(r_typ) :: dstep
  logical :: auto_minmax_dstep
  real(r_typ) :: min_dstep
  real(r_typ) :: max_dstep
  real(r_typ) :: dstep_mult
  logical :: limit_by_local_mesh
  real(r_typ) :: local_mesh_factor
  real(r_typ) :: max_length
  logical :: direction_along_vec_field
  logical :: trace_from_slice_forward
  logical :: trace_from_slice_backward
  
  ! type :: preferences
  !   logical :: cubic_vec_field
  !   logical :: var_dstep
  !   real(r_typ) :: dstep
  !   logical :: auto_minmax_dstep
  !   real(r_typ) :: min_dstep
  !   real(r_typ) :: max_dstep
  !   real(r_typ) :: dstep_mult
  !   logical :: limit_by_local_mesh
  !   real(r_typ) :: local_mesh_factor
  !   real(r_typ) :: max_length
  !   logical :: direction_along_vec_field
  !   logical :: trace_from_slice_forward
  !   logical :: trace_from_slice_backward
  ! end type

  type(preferences) :: prefs

  prefs%cubic_vec_field = cubic_vec_field
  prefs%var_dstep = var_dstep
  prefs%dstep = dstep
  prefs%auto_minmax_dstep = auto_minmax_dstep
  prefs%min_dstep=min_dstep
  prefs%max_dstep = max_dstep
  prefs%dstep_mult = dstep_mult
  prefs%limit_by_local_mesh = limit_by_local_mesh
  prefs%local_mesh_factor = local_mesh_factor
  prefs%max_length = max_length
  prefs%direction_along_vec_field = direction_along_vec_field
  prefs%trace_from_slice_forward = trace_from_slice_forward
  prefs%trace_from_slice_backward = trace_from_slice_backward

  write(*,*) prefs
  ! write(*,*) 'setting visual prefs'
  ! call visual_set_prefs(prefs)

  return
end subroutine init

! subroutine load_b (br_name,bt_name,bp_name)
!   ! use visual
!   use params

!   character(*) :: br_name
!   character(*) :: bt_name
!   character(*) :: bp_name

!   verbose = .true.

!   write (*,*) 'calling visual_setup with', br_name, bt_name, bp_name
!   call visual_setup (br_name,bt_name,bp_name)

! end subroutine load_b

subroutine trace (s0, s1, bs0, bs1, s, traced_to_r_boundary, svec, svec_n)

  use types
  use vars
  use number_types
  use field
  use params
  use debug

  implicit none
  real(r_typ), dimension(3), intent(in) :: s0
  real(r_typ), dimension(3), intent(inout) :: s1, bs0, bs1
  real(r_typ), intent(inout) :: s
  integer, intent(in) :: svec_n
  real(r_typ), intent(inout) :: svec(svec_n, 3)
  logical, intent(inout) :: traced_to_r_boundary
  type(traj) :: xt
  integer :: stride, step_i, svec_step, j, last_step

  call allocate_trajectory_buffer (xt)

  ! type(preferences) :: prefs
  if (verbose.gt.0) then
    write (*,*) 'ready to trace'
    write (*,*) 'cubic:', cubic
    write (*,*) 'b limit 1:', b%lim1(1)
    write (*,*) 'ds variable:', ds%variable
    write (*,*) 'ds%direction:', ds%direction
    write (*,*) 'ds%direction_is_along_b:', ds%direction_is_along_b
    write (*,*) 'size of input svec', size(svec,1)
  end if


  call tracefl (b,ds,s0,s1,bs0,bs1,s, traced_to_r_boundary, xt)

  if (verbose.gt.0) then
    write (*,*) 'traced_to_r_boundary:', traced_to_r_boundary
    write (*,*) 's', s
  end if

! def get_cadence(fl, max_points):
!     '''get the maximum sampling cadence for this field line'''
!     return max(1, int(len(fl)/max_points))

  ! determine if we are taking every nth point to fit the trace in svec.
  stride = max(1, xt%npts/size(svec, 1) + 1)

  if (verbose.gt.0) then
    write (*,*) 'number of trajactory points:', xt%npts
    write (*,*) 'shape of trajectory', shape(xt%x(1)%f)
    write (*,*) 'xt%npts/size(svec, 1):', xt%npts/size(svec, 1)
    write (*,*) 'stride of trajectory', stride
    write (*,*) 'first point', xt%x(1)%f(1), xt%x(2)%f(1), xt%x(3)%f(1)
    write (*,*) 'mod(1, stride)==0', mod(1, stride).eq.0
  end if

  ! Add the first point
  svec_step=1
  if (verbose.gt.0) then
    write(*,*) 'adding first point'
  end if
  do j=1,3
    svec(svec_step, j) = xt%x(j)%f(1)
  enddo
  svec_step=svec_step+1

  ! Add the remaining points
  do step_i=2,xt%npts
    if (svec_step.lt.svec_n.and.mod(step_i, stride).eq.0) then
      do j=1,3
        svec(svec_step, j) = xt%x(j)%f(step_i)
      enddo
      last_step = step_i
      ! write (*,*) 'added point', svec(svec_step, :)
      svec_step=svec_step+1
    endif
  enddo

  ! Add the last point if needed
  if (last_step.lt.xt%npts) then
    do j=1,3
      svec(svec_step, j) = xt%x(j)%f(xt%npts)
    enddo
    if (verbose.gt.0) then
      write (*,*) 'added final point', svec(svec_step,:)
    end if
    svec_step=svec_step+1
  end if

  if (verbose.gt.0) then
    write (*,*) 'svec(final):', svec(svec_step-1,:)
  end if

  ! extra debugging that can be added for pyvisual.
  if (debug_level.ge.2) then
    write (*,*) '*************************************'
    write (*,*) '  xt npts: ', xt%npts, '  stride: ', stride
    write (*,*) '  buffer size: ', size(svec, 1)
    write (*,*) '  last_step (xt): ', last_step
    write (*,*) '  svec npts: ', svec_step-1
    write (*,*) '  s0:         ', s0(1), s0(2), s0(3)
    write (*,*) '  first xt:   ', xt%x(1)%f(1),xt%x(2)%f(1),xt%x(3)%f(1)
    write (*,*) '  first svec: ', svec(1, :)
    if (xt%npts.gt.1) then
      write (*,*) '  2nd xt:     ', xt%x(1)%f(2),xt%x(2)%f(2),xt%x(3)%f(2)
      write (*,*) '  2nd svec:   ', svec(2, :)
      write (*,*) '  xt[n-1]:    ', xt%x(1)%f(xt%npts-1),xt%x(2)%f(xt%npts-1),xt%x(3)%f(xt%npts-1)
      write (*,*) '  svec[n-1]:  ', svec(svec_step-2, :)
    endif
    write (*,*) '  last xt:    ', xt%x(1)%f(xt%npts),xt%x(2)%f(xt%npts),xt%x(3)%f(xt%npts)
    write (*,*) '  last svec:  ', svec(svec_step-1, :)
    write (*,*) '  s1:         ', s1(1), s1(2), s1(3)
    write (*,*) '  traced_to_r_boundary:', traced_to_r_boundary
    write (*,*) '*************************************'
  endif

  call deallocate_trajectory_buffer (xt)


  
  return
end subroutine trace

subroutine print_stats

  use step_size_stats
  use params

  if (verbose.gt.0) then
    stat_ds_avg=0.
    if (stat_n.ne.0) stat_ds_avg=stat_ds_sum/stat_n
    write (*,*)
    write (*,*) '### Field line integration step size statistics:'
    write (*,*) 'Number of field line segments = ',stat_n
    write (*,*) 'Minimum step size used = ',stat_ds_min
    write (*,*) 'Maximum step size used = ',stat_ds_max
    write (*,*) 'Average step size used = ',stat_ds_avg
  end if

end subroutine print_stats


end module mapfl
