c===========================================================================
c
c   This file is part of TISEAN
c 
c   Copyright (c) 1998-2007 Rainer Hegger, Holger Kantz, Thomas Schreiber
c 
c   TISEAN is free software; you can redistribute it and/or modify
c   it under the terms of the GNU General Public License as published by
c   the Free Software Foundation; either version 2 of the License, or
c   (at your option) any later version.
c
c   TISEAN is distributed in the hope that it will be useful,
c   but WITHOUT ANY WARRANTY; without even the implied warranty of
c   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
c   GNU General Public License for more details.
c
c   You should have received a copy of the GNU General Public License
c   along with TISEAN; if not, write to the Free Software
c   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
c
c===========================================================================
c   any_s.f
c   extract numbers from strings
c   author T. Schreiber (1998)
c===========================================================================

      function i_s(s,ierr)
      character*(*) s

      ierr=0
      read(s,'(i20)',err=777) i_s
      if(s.ne.'-'.and.s.ne.'+') return   ! reject a solitary - or +
 777  ierr=1
      end

      function f_s(s,ierr)
      character*(*) s

      ierr=0
      read(s,'(f20.0)',err=777) f_s
      if(s.ne.'-'.and.s.ne.'+') return   ! reject a solitary - or +
 777  ierr=1
      end
