* Problem:    1st_phase
* Class:      MIP
* Rows:       866
* Columns:    254 (200 integer, 180 binary)
* Non-zeros:  2654
* Format:     Free MPS
*
NAME graphdraw-domain
ROWS
 N drawing
 E choose1[Client,CertificateTemplate]
 E choose1[Client,Course]
 E choose1[Client,Role]
 E choose1[Client,User]
 E choose1[Client,CourseSession]
 E choose1[Client,Clinic]
 E choose1[Client,Location]
 E choose1[Client,Therapist]
 E choose1[Client,Instructor]
 E choose1[CertificateTemplate,Course]
 E choose1[CertificateTemplate,Role]
 E choose1[CertificateTemplate,User]
 E choose1[CertificateTemplate,CourseSession]
 E choose1[CertificateTemplate,Clinic]
 E choose1[CertificateTemplate,Location]
 E choose1[CertificateTemplate,Therapist]
 E choose1[CertificateTemplate,Instructor]
 E choose1[Course,Role]
 E choose1[Course,User]
 E choose1[Course,CourseSession]
 E choose1[Course,Clinic]
 E choose1[Course,Location]
 E choose1[Course,Therapist]
 E choose1[Course,Instructor]
 E choose1[Role,User]
 E choose1[Role,CourseSession]
 E choose1[Role,Clinic]
 E choose1[Role,Location]
 E choose1[Role,Therapist]
 E choose1[Role,Instructor]
 E choose1[User,CourseSession]
 E choose1[User,Clinic]
 E choose1[User,Location]
 E choose1[User,Therapist]
 E choose1[User,Instructor]
 E choose1[CourseSession,Clinic]
 E choose1[CourseSession,Location]
 E choose1[CourseSession,Therapist]
 E choose1[CourseSession,Instructor]
 E choose1[Clinic,Location]
 E choose1[Clinic,Therapist]
 E choose1[Clinic,Instructor]
 E choose1[Location,Therapist]
 E choose1[Location,Instructor]
 E choose1[Therapist,Instructor]
 G centerDistAxis1[r,Client]
 G centerDistAxis1[r,CertificateTemplate]
 G centerDistAxis1[r,Course]
 G centerDistAxis1[r,Role]
 G centerDistAxis1[r,User]
 G centerDistAxis1[r,CourseSession]
 G centerDistAxis1[r,Clinic]
 G centerDistAxis1[r,Location]
 G centerDistAxis1[r,Therapist]
 G centerDistAxis1[r,Instructor]
 G centerDistAxis1[c,Client]
 G centerDistAxis1[c,CertificateTemplate]
 G centerDistAxis1[c,Course]
 G centerDistAxis1[c,Role]
 G centerDistAxis1[c,User]
 G centerDistAxis1[c,CourseSession]
 G centerDistAxis1[c,Clinic]
 G centerDistAxis1[c,Location]
 G centerDistAxis1[c,Therapist]
 G centerDistAxis1[c,Instructor]
 G centerDistAxis2[r,Client]
 G centerDistAxis2[r,CertificateTemplate]
 G centerDistAxis2[r,Course]
 G centerDistAxis2[r,Role]
 G centerDistAxis2[r,User]
 G centerDistAxis2[r,CourseSession]
 G centerDistAxis2[r,Clinic]
 G centerDistAxis2[r,Location]
 G centerDistAxis2[r,Therapist]
 G centerDistAxis2[r,Instructor]
 G centerDistAxis2[c,Client]
 G centerDistAxis2[c,CertificateTemplate]
 G centerDistAxis2[c,Course]
 G centerDistAxis2[c,Role]
 G centerDistAxis2[c,User]
 G centerDistAxis2[c,CourseSession]
 G centerDistAxis2[c,Clinic]
 G centerDistAxis2[c,Location]
 G centerDistAxis2[c,Therapist]
 G centerDistAxis2[c,Instructor]
 G axisVdifLB[r,Client,CertificateTemplate]
 G axisVdifLB[r,Client,Course]
 G axisVdifLB[r,Client,Role]
 G axisVdifLB[r,Client,User]
 G axisVdifLB[r,Client,CourseSession]
 G axisVdifLB[r,Client,Clinic]
 G axisVdifLB[r,Client,Location]
 G axisVdifLB[r,Client,Therapist]
 G axisVdifLB[r,Client,Instructor]
 G axisVdifLB[r,CertificateTemplate,Course]
 G axisVdifLB[r,CertificateTemplate,Role]
 G axisVdifLB[r,CertificateTemplate,User]
 G axisVdifLB[r,CertificateTemplate,CourseSession]
 G axisVdifLB[r,CertificateTemplate,Clinic]
 G axisVdifLB[r,CertificateTemplate,Location]
 G axisVdifLB[r,CertificateTemplate,Therapist]
 G axisVdifLB[r,CertificateTemplate,Instructor]
 G axisVdifLB[r,Course,Role]
 G axisVdifLB[r,Course,User]
 G axisVdifLB[r,Course,CourseSession]
 G axisVdifLB[r,Course,Clinic]
 G axisVdifLB[r,Course,Location]
 G axisVdifLB[r,Course,Therapist]
 G axisVdifLB[r,Course,Instructor]
 G axisVdifLB[r,Role,User]
 G axisVdifLB[r,Role,CourseSession]
 G axisVdifLB[r,Role,Clinic]
 G axisVdifLB[r,Role,Location]
 G axisVdifLB[r,Role,Therapist]
 G axisVdifLB[r,Role,Instructor]
 G axisVdifLB[r,User,CourseSession]
 G axisVdifLB[r,User,Clinic]
 G axisVdifLB[r,User,Location]
 G axisVdifLB[r,User,Therapist]
 G axisVdifLB[r,User,Instructor]
 G axisVdifLB[r,CourseSession,Clinic]
 G axisVdifLB[r,CourseSession,Location]
 G axisVdifLB[r,CourseSession,Therapist]
 G axisVdifLB[r,CourseSession,Instructor]
 G axisVdifLB[r,Clinic,Location]
 G axisVdifLB[r,Clinic,Therapist]
 G axisVdifLB[r,Clinic,Instructor]
 G axisVdifLB[r,Location,Therapist]
 G axisVdifLB[r,Location,Instructor]
 G axisVdifLB[r,Therapist,Instructor]
 G axisVdifLB[c,Client,CertificateTemplate]
 G axisVdifLB[c,Client,Course]
 G axisVdifLB[c,Client,Role]
 G axisVdifLB[c,Client,User]
 G axisVdifLB[c,Client,CourseSession]
 G axisVdifLB[c,Client,Clinic]
 G axisVdifLB[c,Client,Location]
 G axisVdifLB[c,Client,Therapist]
 G axisVdifLB[c,Client,Instructor]
 G axisVdifLB[c,CertificateTemplate,Course]
 G axisVdifLB[c,CertificateTemplate,Role]
 G axisVdifLB[c,CertificateTemplate,User]
 G axisVdifLB[c,CertificateTemplate,CourseSession]
 G axisVdifLB[c,CertificateTemplate,Clinic]
 G axisVdifLB[c,CertificateTemplate,Location]
 G axisVdifLB[c,CertificateTemplate,Therapist]
 G axisVdifLB[c,CertificateTemplate,Instructor]
 G axisVdifLB[c,Course,Role]
 G axisVdifLB[c,Course,User]
 G axisVdifLB[c,Course,CourseSession]
 G axisVdifLB[c,Course,Clinic]
 G axisVdifLB[c,Course,Location]
 G axisVdifLB[c,Course,Therapist]
 G axisVdifLB[c,Course,Instructor]
 G axisVdifLB[c,Role,User]
 G axisVdifLB[c,Role,CourseSession]
 G axisVdifLB[c,Role,Clinic]
 G axisVdifLB[c,Role,Location]
 G axisVdifLB[c,Role,Therapist]
 G axisVdifLB[c,Role,Instructor]
 G axisVdifLB[c,User,CourseSession]
 G axisVdifLB[c,User,Clinic]
 G axisVdifLB[c,User,Location]
 G axisVdifLB[c,User,Therapist]
 G axisVdifLB[c,User,Instructor]
 G axisVdifLB[c,CourseSession,Clinic]
 G axisVdifLB[c,CourseSession,Location]
 G axisVdifLB[c,CourseSession,Therapist]
 G axisVdifLB[c,CourseSession,Instructor]
 G axisVdifLB[c,Clinic,Location]
 G axisVdifLB[c,Clinic,Therapist]
 G axisVdifLB[c,Clinic,Instructor]
 G axisVdifLB[c,Location,Therapist]
 G axisVdifLB[c,Location,Instructor]
 G axisVdifLB[c,Therapist,Instructor]
 G axisVdifRT[r,Client,CertificateTemplate]
 G axisVdifRT[r,Client,Course]
 G axisVdifRT[r,Client,Role]
 G axisVdifRT[r,Client,User]
 G axisVdifRT[r,Client,CourseSession]
 G axisVdifRT[r,Client,Clinic]
 G axisVdifRT[r,Client,Location]
 G axisVdifRT[r,Client,Therapist]
 G axisVdifRT[r,Client,Instructor]
 G axisVdifRT[r,CertificateTemplate,Course]
 G axisVdifRT[r,CertificateTemplate,Role]
 G axisVdifRT[r,CertificateTemplate,User]
 G axisVdifRT[r,CertificateTemplate,CourseSession]
 G axisVdifRT[r,CertificateTemplate,Clinic]
 G axisVdifRT[r,CertificateTemplate,Location]
 G axisVdifRT[r,CertificateTemplate,Therapist]
 G axisVdifRT[r,CertificateTemplate,Instructor]
 G axisVdifRT[r,Course,Role]
 G axisVdifRT[r,Course,User]
 G axisVdifRT[r,Course,CourseSession]
 G axisVdifRT[r,Course,Clinic]
 G axisVdifRT[r,Course,Location]
 G axisVdifRT[r,Course,Therapist]
 G axisVdifRT[r,Course,Instructor]
 G axisVdifRT[r,Role,User]
 G axisVdifRT[r,Role,CourseSession]
 G axisVdifRT[r,Role,Clinic]
 G axisVdifRT[r,Role,Location]
 G axisVdifRT[r,Role,Therapist]
 G axisVdifRT[r,Role,Instructor]
 G axisVdifRT[r,User,CourseSession]
 G axisVdifRT[r,User,Clinic]
 G axisVdifRT[r,User,Location]
 G axisVdifRT[r,User,Therapist]
 G axisVdifRT[r,User,Instructor]
 G axisVdifRT[r,CourseSession,Clinic]
 G axisVdifRT[r,CourseSession,Location]
 G axisVdifRT[r,CourseSession,Therapist]
 G axisVdifRT[r,CourseSession,Instructor]
 G axisVdifRT[r,Clinic,Location]
 G axisVdifRT[r,Clinic,Therapist]
 G axisVdifRT[r,Clinic,Instructor]
 G axisVdifRT[r,Location,Therapist]
 G axisVdifRT[r,Location,Instructor]
 G axisVdifRT[r,Therapist,Instructor]
 G axisVdifRT[c,Client,CertificateTemplate]
 G axisVdifRT[c,Client,Course]
 G axisVdifRT[c,Client,Role]
 G axisVdifRT[c,Client,User]
 G axisVdifRT[c,Client,CourseSession]
 G axisVdifRT[c,Client,Clinic]
 G axisVdifRT[c,Client,Location]
 G axisVdifRT[c,Client,Therapist]
 G axisVdifRT[c,Client,Instructor]
 G axisVdifRT[c,CertificateTemplate,Course]
 G axisVdifRT[c,CertificateTemplate,Role]
 G axisVdifRT[c,CertificateTemplate,User]
 G axisVdifRT[c,CertificateTemplate,CourseSession]
 G axisVdifRT[c,CertificateTemplate,Clinic]
 G axisVdifRT[c,CertificateTemplate,Location]
 G axisVdifRT[c,CertificateTemplate,Therapist]
 G axisVdifRT[c,CertificateTemplate,Instructor]
 G axisVdifRT[c,Course,Role]
 G axisVdifRT[c,Course,User]
 G axisVdifRT[c,Course,CourseSession]
 G axisVdifRT[c,Course,Clinic]
 G axisVdifRT[c,Course,Location]
 G axisVdifRT[c,Course,Therapist]
 G axisVdifRT[c,Course,Instructor]
 G axisVdifRT[c,Role,User]
 G axisVdifRT[c,Role,CourseSession]
 G axisVdifRT[c,Role,Clinic]
 G axisVdifRT[c,Role,Location]
 G axisVdifRT[c,Role,Therapist]
 G axisVdifRT[c,Role,Instructor]
 G axisVdifRT[c,User,CourseSession]
 G axisVdifRT[c,User,Clinic]
 G axisVdifRT[c,User,Location]
 G axisVdifRT[c,User,Therapist]
 G axisVdifRT[c,User,Instructor]
 G axisVdifRT[c,CourseSession,Clinic]
 G axisVdifRT[c,CourseSession,Location]
 G axisVdifRT[c,CourseSession,Therapist]
 G axisVdifRT[c,CourseSession,Instructor]
 G axisVdifRT[c,Clinic,Location]
 G axisVdifRT[c,Clinic,Therapist]
 G axisVdifRT[c,Clinic,Instructor]
 G axisVdifRT[c,Location,Therapist]
 G axisVdifRT[c,Location,Instructor]
 G axisVdifRT[c,Therapist,Instructor]
 G compDistAxis1[r,Client,CertificateTemplate]
 G compDistAxis1[r,Client,CourseSession]
 G compDistAxis1[r,Client,Instructor]
 G compDistAxis1[r,Client,Location]
 G compDistAxis1[r,Client,User]
 G compDistAxis1[r,CertificateTemplate,CourseSession]
 G compDistAxis1[r,Course,CourseSession]
 G compDistAxis1[r,CourseSession,Instructor]
 G compDistAxis1[r,Instructor,Location]
 G compDistAxis1[r,Clinic,Location]
 G compDistAxis1[r,Clinic,Therapist]
 G compDistAxis1[r,Location,Therapist]
 G compDistAxis1[r,Location,CourseSession]
 G compDistAxis1[r,Therapist,CourseSession]
 G compDistAxis1[r,Role,User]
 G compDistAxis1[r,User,Instructor]
 G compDistAxis1[r,User,Therapist]
 G compDistAxis1[c,Client,CertificateTemplate]
 G compDistAxis1[c,Client,CourseSession]
 G compDistAxis1[c,Client,Instructor]
 G compDistAxis1[c,Client,Location]
 G compDistAxis1[c,Client,User]
 G compDistAxis1[c,CertificateTemplate,CourseSession]
 G compDistAxis1[c,Course,CourseSession]
 G compDistAxis1[c,CourseSession,Instructor]
 G compDistAxis1[c,Instructor,Location]
 G compDistAxis1[c,Clinic,Location]
 G compDistAxis1[c,Clinic,Therapist]
 G compDistAxis1[c,Location,Therapist]
 G compDistAxis1[c,Location,CourseSession]
 G compDistAxis1[c,Therapist,CourseSession]
 G compDistAxis1[c,Role,User]
 G compDistAxis1[c,User,Instructor]
 G compDistAxis1[c,User,Therapist]
 G compDistAxis2[r,Client,CertificateTemplate]
 G compDistAxis2[r,Client,CourseSession]
 G compDistAxis2[r,Client,Instructor]
 G compDistAxis2[r,Client,Location]
 G compDistAxis2[r,Client,User]
 G compDistAxis2[r,CertificateTemplate,CourseSession]
 G compDistAxis2[r,Course,CourseSession]
 G compDistAxis2[r,CourseSession,Instructor]
 G compDistAxis2[r,Instructor,Location]
 G compDistAxis2[r,Clinic,Location]
 G compDistAxis2[r,Clinic,Therapist]
 G compDistAxis2[r,Location,Therapist]
 G compDistAxis2[r,Location,CourseSession]
 G compDistAxis2[r,Therapist,CourseSession]
 G compDistAxis2[r,Role,User]
 G compDistAxis2[r,User,Instructor]
 G compDistAxis2[r,User,Therapist]
 G compDistAxis2[c,Client,CertificateTemplate]
 G compDistAxis2[c,Client,CourseSession]
 G compDistAxis2[c,Client,Instructor]
 G compDistAxis2[c,Client,Location]
 G compDistAxis2[c,Client,User]
 G compDistAxis2[c,CertificateTemplate,CourseSession]
 G compDistAxis2[c,Course,CourseSession]
 G compDistAxis2[c,CourseSession,Instructor]
 G compDistAxis2[c,Instructor,Location]
 G compDistAxis2[c,Clinic,Location]
 G compDistAxis2[c,Clinic,Therapist]
 G compDistAxis2[c,Location,Therapist]
 G compDistAxis2[c,Location,CourseSession]
 G compDistAxis2[c,Therapist,CourseSession]
 G compDistAxis2[c,Role,User]
 G compDistAxis2[c,User,Instructor]
 G compDistAxis2[c,User,Therapist]
 G DistSumLB[Client,CertificateTemplate]
 G DistSumLB[Client,CourseSession]
 G DistSumLB[Client,Instructor]
 G DistSumLB[Client,Location]
 G DistSumLB[Client,User]
 G DistSumLB[CertificateTemplate,CourseSession]
 G DistSumLB[Course,CourseSession]
 G DistSumLB[CourseSession,Instructor]
 G DistSumLB[Instructor,Location]
 G DistSumLB[Clinic,Location]
 G DistSumLB[Clinic,Therapist]
 G DistSumLB[Location,Therapist]
 G DistSumLB[Location,CourseSession]
 G DistSumLB[Therapist,CourseSession]
 G DistSumLB[Role,User]
 G DistSumLB[User,Instructor]
 G DistSumLB[User,Therapist]
 G CenterDistSumLB
 G DistAxisLB1[r,Client,CertificateTemplate]
 G DistAxisLB1[r,Client,CourseSession]
 G DistAxisLB1[r,Client,Instructor]
 G DistAxisLB1[r,Client,Location]
 G DistAxisLB1[r,Client,User]
 G DistAxisLB1[r,CertificateTemplate,CourseSession]
 G DistAxisLB1[r,Course,CourseSession]
 G DistAxisLB1[r,CourseSession,Instructor]
 G DistAxisLB1[r,Instructor,Location]
 G DistAxisLB1[r,Clinic,Location]
 G DistAxisLB1[r,Clinic,Therapist]
 G DistAxisLB1[r,Location,Therapist]
 G DistAxisLB1[r,Location,CourseSession]
 G DistAxisLB1[r,Therapist,CourseSession]
 G DistAxisLB1[r,Role,User]
 G DistAxisLB1[r,User,Instructor]
 G DistAxisLB1[r,User,Therapist]
 G DistAxisLB1[c,Client,CertificateTemplate]
 G DistAxisLB1[c,Client,CourseSession]
 G DistAxisLB1[c,Client,Instructor]
 G DistAxisLB1[c,Client,Location]
 G DistAxisLB1[c,Client,User]
 G DistAxisLB1[c,CertificateTemplate,CourseSession]
 G DistAxisLB1[c,Course,CourseSession]
 G DistAxisLB1[c,CourseSession,Instructor]
 G DistAxisLB1[c,Instructor,Location]
 G DistAxisLB1[c,Clinic,Location]
 G DistAxisLB1[c,Clinic,Therapist]
 G DistAxisLB1[c,Location,Therapist]
 G DistAxisLB1[c,Location,CourseSession]
 G DistAxisLB1[c,Therapist,CourseSession]
 G DistAxisLB1[c,Role,User]
 G DistAxisLB1[c,User,Instructor]
 G DistAxisLB1[c,User,Therapist]
 L triangle1[Client,CertificateTemplate,Course,r]
 L triangle1[Client,CertificateTemplate,Course,c]
 L triangle1[Client,CertificateTemplate,Role,r]
 L triangle1[Client,CertificateTemplate,Role,c]
 L triangle1[Client,CertificateTemplate,User,r]
 L triangle1[Client,CertificateTemplate,User,c]
 L triangle1[Client,CertificateTemplate,CourseSession,r]
 L triangle1[Client,CertificateTemplate,CourseSession,c]
 L triangle1[Client,CertificateTemplate,Clinic,r]
 L triangle1[Client,CertificateTemplate,Clinic,c]
 L triangle1[Client,CertificateTemplate,Location,r]
 L triangle1[Client,CertificateTemplate,Location,c]
 L triangle1[Client,CertificateTemplate,Therapist,r]
 L triangle1[Client,CertificateTemplate,Therapist,c]
 L triangle1[Client,CertificateTemplate,Instructor,r]
 L triangle1[Client,CertificateTemplate,Instructor,c]
 L triangle1[Client,Course,Role,r]
 L triangle1[Client,Course,Role,c]
 L triangle1[Client,Course,User,r]
 L triangle1[Client,Course,User,c]
 L triangle1[Client,Course,CourseSession,r]
 L triangle1[Client,Course,CourseSession,c]
 L triangle1[Client,Course,Clinic,r]
 L triangle1[Client,Course,Clinic,c]
 L triangle1[Client,Course,Location,r]
 L triangle1[Client,Course,Location,c]
 L triangle1[Client,Course,Therapist,r]
 L triangle1[Client,Course,Therapist,c]
 L triangle1[Client,Course,Instructor,r]
 L triangle1[Client,Course,Instructor,c]
 L triangle1[Client,Role,User,r]
 L triangle1[Client,Role,User,c]
 L triangle1[Client,Role,CourseSession,r]
 L triangle1[Client,Role,CourseSession,c]
 L triangle1[Client,Role,Clinic,r]
 L triangle1[Client,Role,Clinic,c]
 L triangle1[Client,Role,Location,r]
 L triangle1[Client,Role,Location,c]
 L triangle1[Client,Role,Therapist,r]
 L triangle1[Client,Role,Therapist,c]
 L triangle1[Client,Role,Instructor,r]
 L triangle1[Client,Role,Instructor,c]
 L triangle1[Client,User,CourseSession,r]
 L triangle1[Client,User,CourseSession,c]
 L triangle1[Client,User,Clinic,r]
 L triangle1[Client,User,Clinic,c]
 L triangle1[Client,User,Location,r]
 L triangle1[Client,User,Location,c]
 L triangle1[Client,User,Therapist,r]
 L triangle1[Client,User,Therapist,c]
 L triangle1[Client,User,Instructor,r]
 L triangle1[Client,User,Instructor,c]
 L triangle1[Client,CourseSession,Clinic,r]
 L triangle1[Client,CourseSession,Clinic,c]
 L triangle1[Client,CourseSession,Location,r]
 L triangle1[Client,CourseSession,Location,c]
 L triangle1[Client,CourseSession,Therapist,r]
 L triangle1[Client,CourseSession,Therapist,c]
 L triangle1[Client,CourseSession,Instructor,r]
 L triangle1[Client,CourseSession,Instructor,c]
 L triangle1[Client,Clinic,Location,r]
 L triangle1[Client,Clinic,Location,c]
 L triangle1[Client,Clinic,Therapist,r]
 L triangle1[Client,Clinic,Therapist,c]
 L triangle1[Client,Clinic,Instructor,r]
 L triangle1[Client,Clinic,Instructor,c]
 L triangle1[Client,Location,Therapist,r]
 L triangle1[Client,Location,Therapist,c]
 L triangle1[Client,Location,Instructor,r]
 L triangle1[Client,Location,Instructor,c]
 L triangle1[Client,Therapist,Instructor,r]
 L triangle1[Client,Therapist,Instructor,c]
 L triangle1[CertificateTemplate,Course,Role,r]
 L triangle1[CertificateTemplate,Course,Role,c]
 L triangle1[CertificateTemplate,Course,User,r]
 L triangle1[CertificateTemplate,Course,User,c]
 L triangle1[CertificateTemplate,Course,CourseSession,r]
 L triangle1[CertificateTemplate,Course,CourseSession,c]
 L triangle1[CertificateTemplate,Course,Clinic,r]
 L triangle1[CertificateTemplate,Course,Clinic,c]
 L triangle1[CertificateTemplate,Course,Location,r]
 L triangle1[CertificateTemplate,Course,Location,c]
 L triangle1[CertificateTemplate,Course,Therapist,r]
 L triangle1[CertificateTemplate,Course,Therapist,c]
 L triangle1[CertificateTemplate,Course,Instructor,r]
 L triangle1[CertificateTemplate,Course,Instructor,c]
 L triangle1[CertificateTemplate,Role,User,r]
 L triangle1[CertificateTemplate,Role,User,c]
 L triangle1[CertificateTemplate,Role,CourseSession,r]
 L triangle1[CertificateTemplate,Role,CourseSession,c]
 L triangle1[CertificateTemplate,Role,Clinic,r]
 L triangle1[CertificateTemplate,Role,Clinic,c]
 L triangle1[CertificateTemplate,Role,Location,r]
 L triangle1[CertificateTemplate,Role,Location,c]
 L triangle1[CertificateTemplate,Role,Therapist,r]
 L triangle1[CertificateTemplate,Role,Therapist,c]
 L triangle1[CertificateTemplate,Role,Instructor,r]
 L triangle1[CertificateTemplate,Role,Instructor,c]
 L triangle1[CertificateTemplate,User,CourseSession,r]
 L triangle1[CertificateTemplate,User,CourseSession,c]
 L triangle1[CertificateTemplate,User,Clinic,r]
 L triangle1[CertificateTemplate,User,Clinic,c]
 L triangle1[CertificateTemplate,User,Location,r]
 L triangle1[CertificateTemplate,User,Location,c]
 L triangle1[CertificateTemplate,User,Therapist,r]
 L triangle1[CertificateTemplate,User,Therapist,c]
 L triangle1[CertificateTemplate,User,Instructor,r]
 L triangle1[CertificateTemplate,User,Instructor,c]
 L triangle1[CertificateTemplate,CourseSession,Clinic,r]
 L triangle1[CertificateTemplate,CourseSession,Clinic,c]
 L triangle1[CertificateTemplate,CourseSession,Location,r]
 L triangle1[CertificateTemplate,CourseSession,Location,c]
 L triangle1[CertificateTemplate,CourseSession,Therapist,r]
 L triangle1[CertificateTemplate,CourseSession,Therapist,c]
 L triangle1[CertificateTemplate,CourseSession,Instructor,r]
 L triangle1[CertificateTemplate,CourseSession,Instructor,c]
 L triangle1[CertificateTemplate,Clinic,Location,r]
 L triangle1[CertificateTemplate,Clinic,Location,c]
 L triangle1[CertificateTemplate,Clinic,Therapist,r]
 L triangle1[CertificateTemplate,Clinic,Therapist,c]
 L triangle1[CertificateTemplate,Clinic,Instructor,r]
 L triangle1[CertificateTemplate,Clinic,Instructor,c]
 L triangle1[CertificateTemplate,Location,Therapist,r]
 L triangle1[CertificateTemplate,Location,Therapist,c]
 L triangle1[CertificateTemplate,Location,Instructor,r]
 L triangle1[CertificateTemplate,Location,Instructor,c]
 L triangle1[CertificateTemplate,Therapist,Instructor,r]
 L triangle1[CertificateTemplate,Therapist,Instructor,c]
 L triangle1[Course,Role,User,r]
 L triangle1[Course,Role,User,c]
 L triangle1[Course,Role,CourseSession,r]
 L triangle1[Course,Role,CourseSession,c]
 L triangle1[Course,Role,Clinic,r]
 L triangle1[Course,Role,Clinic,c]
 L triangle1[Course,Role,Location,r]
 L triangle1[Course,Role,Location,c]
 L triangle1[Course,Role,Therapist,r]
 L triangle1[Course,Role,Therapist,c]
 L triangle1[Course,Role,Instructor,r]
 L triangle1[Course,Role,Instructor,c]
 L triangle1[Course,User,CourseSession,r]
 L triangle1[Course,User,CourseSession,c]
 L triangle1[Course,User,Clinic,r]
 L triangle1[Course,User,Clinic,c]
 L triangle1[Course,User,Location,r]
 L triangle1[Course,User,Location,c]
 L triangle1[Course,User,Therapist,r]
 L triangle1[Course,User,Therapist,c]
 L triangle1[Course,User,Instructor,r]
 L triangle1[Course,User,Instructor,c]
 L triangle1[Course,CourseSession,Clinic,r]
 L triangle1[Course,CourseSession,Clinic,c]
 L triangle1[Course,CourseSession,Location,r]
 L triangle1[Course,CourseSession,Location,c]
 L triangle1[Course,CourseSession,Therapist,r]
 L triangle1[Course,CourseSession,Therapist,c]
 L triangle1[Course,CourseSession,Instructor,r]
 L triangle1[Course,CourseSession,Instructor,c]
 L triangle1[Course,Clinic,Location,r]
 L triangle1[Course,Clinic,Location,c]
 L triangle1[Course,Clinic,Therapist,r]
 L triangle1[Course,Clinic,Therapist,c]
 L triangle1[Course,Clinic,Instructor,r]
 L triangle1[Course,Clinic,Instructor,c]
 L triangle1[Course,Location,Therapist,r]
 L triangle1[Course,Location,Therapist,c]
 L triangle1[Course,Location,Instructor,r]
 L triangle1[Course,Location,Instructor,c]
 L triangle1[Course,Therapist,Instructor,r]
 L triangle1[Course,Therapist,Instructor,c]
 L triangle1[Role,User,CourseSession,r]
 L triangle1[Role,User,CourseSession,c]
 L triangle1[Role,User,Clinic,r]
 L triangle1[Role,User,Clinic,c]
 L triangle1[Role,User,Location,r]
 L triangle1[Role,User,Location,c]
 L triangle1[Role,User,Therapist,r]
 L triangle1[Role,User,Therapist,c]
 L triangle1[Role,User,Instructor,r]
 L triangle1[Role,User,Instructor,c]
 L triangle1[Role,CourseSession,Clinic,r]
 L triangle1[Role,CourseSession,Clinic,c]
 L triangle1[Role,CourseSession,Location,r]
 L triangle1[Role,CourseSession,Location,c]
 L triangle1[Role,CourseSession,Therapist,r]
 L triangle1[Role,CourseSession,Therapist,c]
 L triangle1[Role,CourseSession,Instructor,r]
 L triangle1[Role,CourseSession,Instructor,c]
 L triangle1[Role,Clinic,Location,r]
 L triangle1[Role,Clinic,Location,c]
 L triangle1[Role,Clinic,Therapist,r]
 L triangle1[Role,Clinic,Therapist,c]
 L triangle1[Role,Clinic,Instructor,r]
 L triangle1[Role,Clinic,Instructor,c]
 L triangle1[Role,Location,Therapist,r]
 L triangle1[Role,Location,Therapist,c]
 L triangle1[Role,Location,Instructor,r]
 L triangle1[Role,Location,Instructor,c]
 L triangle1[Role,Therapist,Instructor,r]
 L triangle1[Role,Therapist,Instructor,c]
 L triangle1[User,CourseSession,Clinic,r]
 L triangle1[User,CourseSession,Clinic,c]
 L triangle1[User,CourseSession,Location,r]
 L triangle1[User,CourseSession,Location,c]
 L triangle1[User,CourseSession,Therapist,r]
 L triangle1[User,CourseSession,Therapist,c]
 L triangle1[User,CourseSession,Instructor,r]
 L triangle1[User,CourseSession,Instructor,c]
 L triangle1[User,Clinic,Location,r]
 L triangle1[User,Clinic,Location,c]
 L triangle1[User,Clinic,Therapist,r]
 L triangle1[User,Clinic,Therapist,c]
 L triangle1[User,Clinic,Instructor,r]
 L triangle1[User,Clinic,Instructor,c]
 L triangle1[User,Location,Therapist,r]
 L triangle1[User,Location,Therapist,c]
 L triangle1[User,Location,Instructor,r]
 L triangle1[User,Location,Instructor,c]
 L triangle1[User,Therapist,Instructor,r]
 L triangle1[User,Therapist,Instructor,c]
 L triangle1[CourseSession,Clinic,Location,r]
 L triangle1[CourseSession,Clinic,Location,c]
 L triangle1[CourseSession,Clinic,Therapist,r]
 L triangle1[CourseSession,Clinic,Therapist,c]
 L triangle1[CourseSession,Clinic,Instructor,r]
 L triangle1[CourseSession,Clinic,Instructor,c]
 L triangle1[CourseSession,Location,Therapist,r]
 L triangle1[CourseSession,Location,Therapist,c]
 L triangle1[CourseSession,Location,Instructor,r]
 L triangle1[CourseSession,Location,Instructor,c]
 L triangle1[CourseSession,Therapist,Instructor,r]
 L triangle1[CourseSession,Therapist,Instructor,c]
 L triangle1[Clinic,Location,Therapist,r]
 L triangle1[Clinic,Location,Therapist,c]
 L triangle1[Clinic,Location,Instructor,r]
 L triangle1[Clinic,Location,Instructor,c]
 L triangle1[Clinic,Therapist,Instructor,r]
 L triangle1[Clinic,Therapist,Instructor,c]
 L triangle1[Location,Therapist,Instructor,r]
 L triangle1[Location,Therapist,Instructor,c]
 L triangle2[Client,CertificateTemplate,Course,r]
 L triangle2[Client,CertificateTemplate,Course,c]
 L triangle2[Client,CertificateTemplate,Role,r]
 L triangle2[Client,CertificateTemplate,Role,c]
 L triangle2[Client,CertificateTemplate,User,r]
 L triangle2[Client,CertificateTemplate,User,c]
 L triangle2[Client,CertificateTemplate,CourseSession,r]
 L triangle2[Client,CertificateTemplate,CourseSession,c]
 L triangle2[Client,CertificateTemplate,Clinic,r]
 L triangle2[Client,CertificateTemplate,Clinic,c]
 L triangle2[Client,CertificateTemplate,Location,r]
 L triangle2[Client,CertificateTemplate,Location,c]
 L triangle2[Client,CertificateTemplate,Therapist,r]
 L triangle2[Client,CertificateTemplate,Therapist,c]
 L triangle2[Client,CertificateTemplate,Instructor,r]
 L triangle2[Client,CertificateTemplate,Instructor,c]
 L triangle2[Client,Course,Role,r]
 L triangle2[Client,Course,Role,c]
 L triangle2[Client,Course,User,r]
 L triangle2[Client,Course,User,c]
 L triangle2[Client,Course,CourseSession,r]
 L triangle2[Client,Course,CourseSession,c]
 L triangle2[Client,Course,Clinic,r]
 L triangle2[Client,Course,Clinic,c]
 L triangle2[Client,Course,Location,r]
 L triangle2[Client,Course,Location,c]
 L triangle2[Client,Course,Therapist,r]
 L triangle2[Client,Course,Therapist,c]
 L triangle2[Client,Course,Instructor,r]
 L triangle2[Client,Course,Instructor,c]
 L triangle2[Client,Role,User,r]
 L triangle2[Client,Role,User,c]
 L triangle2[Client,Role,CourseSession,r]
 L triangle2[Client,Role,CourseSession,c]
 L triangle2[Client,Role,Clinic,r]
 L triangle2[Client,Role,Clinic,c]
 L triangle2[Client,Role,Location,r]
 L triangle2[Client,Role,Location,c]
 L triangle2[Client,Role,Therapist,r]
 L triangle2[Client,Role,Therapist,c]
 L triangle2[Client,Role,Instructor,r]
 L triangle2[Client,Role,Instructor,c]
 L triangle2[Client,User,CourseSession,r]
 L triangle2[Client,User,CourseSession,c]
 L triangle2[Client,User,Clinic,r]
 L triangle2[Client,User,Clinic,c]
 L triangle2[Client,User,Location,r]
 L triangle2[Client,User,Location,c]
 L triangle2[Client,User,Therapist,r]
 L triangle2[Client,User,Therapist,c]
 L triangle2[Client,User,Instructor,r]
 L triangle2[Client,User,Instructor,c]
 L triangle2[Client,CourseSession,Clinic,r]
 L triangle2[Client,CourseSession,Clinic,c]
 L triangle2[Client,CourseSession,Location,r]
 L triangle2[Client,CourseSession,Location,c]
 L triangle2[Client,CourseSession,Therapist,r]
 L triangle2[Client,CourseSession,Therapist,c]
 L triangle2[Client,CourseSession,Instructor,r]
 L triangle2[Client,CourseSession,Instructor,c]
 L triangle2[Client,Clinic,Location,r]
 L triangle2[Client,Clinic,Location,c]
 L triangle2[Client,Clinic,Therapist,r]
 L triangle2[Client,Clinic,Therapist,c]
 L triangle2[Client,Clinic,Instructor,r]
 L triangle2[Client,Clinic,Instructor,c]
 L triangle2[Client,Location,Therapist,r]
 L triangle2[Client,Location,Therapist,c]
 L triangle2[Client,Location,Instructor,r]
 L triangle2[Client,Location,Instructor,c]
 L triangle2[Client,Therapist,Instructor,r]
 L triangle2[Client,Therapist,Instructor,c]
 L triangle2[CertificateTemplate,Course,Role,r]
 L triangle2[CertificateTemplate,Course,Role,c]
 L triangle2[CertificateTemplate,Course,User,r]
 L triangle2[CertificateTemplate,Course,User,c]
 L triangle2[CertificateTemplate,Course,CourseSession,r]
 L triangle2[CertificateTemplate,Course,CourseSession,c]
 L triangle2[CertificateTemplate,Course,Clinic,r]
 L triangle2[CertificateTemplate,Course,Clinic,c]
 L triangle2[CertificateTemplate,Course,Location,r]
 L triangle2[CertificateTemplate,Course,Location,c]
 L triangle2[CertificateTemplate,Course,Therapist,r]
 L triangle2[CertificateTemplate,Course,Therapist,c]
 L triangle2[CertificateTemplate,Course,Instructor,r]
 L triangle2[CertificateTemplate,Course,Instructor,c]
 L triangle2[CertificateTemplate,Role,User,r]
 L triangle2[CertificateTemplate,Role,User,c]
 L triangle2[CertificateTemplate,Role,CourseSession,r]
 L triangle2[CertificateTemplate,Role,CourseSession,c]
 L triangle2[CertificateTemplate,Role,Clinic,r]
 L triangle2[CertificateTemplate,Role,Clinic,c]
 L triangle2[CertificateTemplate,Role,Location,r]
 L triangle2[CertificateTemplate,Role,Location,c]
 L triangle2[CertificateTemplate,Role,Therapist,r]
 L triangle2[CertificateTemplate,Role,Therapist,c]
 L triangle2[CertificateTemplate,Role,Instructor,r]
 L triangle2[CertificateTemplate,Role,Instructor,c]
 L triangle2[CertificateTemplate,User,CourseSession,r]
 L triangle2[CertificateTemplate,User,CourseSession,c]
 L triangle2[CertificateTemplate,User,Clinic,r]
 L triangle2[CertificateTemplate,User,Clinic,c]
 L triangle2[CertificateTemplate,User,Location,r]
 L triangle2[CertificateTemplate,User,Location,c]
 L triangle2[CertificateTemplate,User,Therapist,r]
 L triangle2[CertificateTemplate,User,Therapist,c]
 L triangle2[CertificateTemplate,User,Instructor,r]
 L triangle2[CertificateTemplate,User,Instructor,c]
 L triangle2[CertificateTemplate,CourseSession,Clinic,r]
 L triangle2[CertificateTemplate,CourseSession,Clinic,c]
 L triangle2[CertificateTemplate,CourseSession,Location,r]
 L triangle2[CertificateTemplate,CourseSession,Location,c]
 L triangle2[CertificateTemplate,CourseSession,Therapist,r]
 L triangle2[CertificateTemplate,CourseSession,Therapist,c]
 L triangle2[CertificateTemplate,CourseSession,Instructor,r]
 L triangle2[CertificateTemplate,CourseSession,Instructor,c]
 L triangle2[CertificateTemplate,Clinic,Location,r]
 L triangle2[CertificateTemplate,Clinic,Location,c]
 L triangle2[CertificateTemplate,Clinic,Therapist,r]
 L triangle2[CertificateTemplate,Clinic,Therapist,c]
 L triangle2[CertificateTemplate,Clinic,Instructor,r]
 L triangle2[CertificateTemplate,Clinic,Instructor,c]
 L triangle2[CertificateTemplate,Location,Therapist,r]
 L triangle2[CertificateTemplate,Location,Therapist,c]
 L triangle2[CertificateTemplate,Location,Instructor,r]
 L triangle2[CertificateTemplate,Location,Instructor,c]
 L triangle2[CertificateTemplate,Therapist,Instructor,r]
 L triangle2[CertificateTemplate,Therapist,Instructor,c]
 L triangle2[Course,Role,User,r]
 L triangle2[Course,Role,User,c]
 L triangle2[Course,Role,CourseSession,r]
 L triangle2[Course,Role,CourseSession,c]
 L triangle2[Course,Role,Clinic,r]
 L triangle2[Course,Role,Clinic,c]
 L triangle2[Course,Role,Location,r]
 L triangle2[Course,Role,Location,c]
 L triangle2[Course,Role,Therapist,r]
 L triangle2[Course,Role,Therapist,c]
 L triangle2[Course,Role,Instructor,r]
 L triangle2[Course,Role,Instructor,c]
 L triangle2[Course,User,CourseSession,r]
 L triangle2[Course,User,CourseSession,c]
 L triangle2[Course,User,Clinic,r]
 L triangle2[Course,User,Clinic,c]
 L triangle2[Course,User,Location,r]
 L triangle2[Course,User,Location,c]
 L triangle2[Course,User,Therapist,r]
 L triangle2[Course,User,Therapist,c]
 L triangle2[Course,User,Instructor,r]
 L triangle2[Course,User,Instructor,c]
 L triangle2[Course,CourseSession,Clinic,r]
 L triangle2[Course,CourseSession,Clinic,c]
 L triangle2[Course,CourseSession,Location,r]
 L triangle2[Course,CourseSession,Location,c]
 L triangle2[Course,CourseSession,Therapist,r]
 L triangle2[Course,CourseSession,Therapist,c]
 L triangle2[Course,CourseSession,Instructor,r]
 L triangle2[Course,CourseSession,Instructor,c]
 L triangle2[Course,Clinic,Location,r]
 L triangle2[Course,Clinic,Location,c]
 L triangle2[Course,Clinic,Therapist,r]
 L triangle2[Course,Clinic,Therapist,c]
 L triangle2[Course,Clinic,Instructor,r]
 L triangle2[Course,Clinic,Instructor,c]
 L triangle2[Course,Location,Therapist,r]
 L triangle2[Course,Location,Therapist,c]
 L triangle2[Course,Location,Instructor,r]
 L triangle2[Course,Location,Instructor,c]
 L triangle2[Course,Therapist,Instructor,r]
 L triangle2[Course,Therapist,Instructor,c]
 L triangle2[Role,User,CourseSession,r]
 L triangle2[Role,User,CourseSession,c]
 L triangle2[Role,User,Clinic,r]
 L triangle2[Role,User,Clinic,c]
 L triangle2[Role,User,Location,r]
 L triangle2[Role,User,Location,c]
 L triangle2[Role,User,Therapist,r]
 L triangle2[Role,User,Therapist,c]
 L triangle2[Role,User,Instructor,r]
 L triangle2[Role,User,Instructor,c]
 L triangle2[Role,CourseSession,Clinic,r]
 L triangle2[Role,CourseSession,Clinic,c]
 L triangle2[Role,CourseSession,Location,r]
 L triangle2[Role,CourseSession,Location,c]
 L triangle2[Role,CourseSession,Therapist,r]
 L triangle2[Role,CourseSession,Therapist,c]
 L triangle2[Role,CourseSession,Instructor,r]
 L triangle2[Role,CourseSession,Instructor,c]
 L triangle2[Role,Clinic,Location,r]
 L triangle2[Role,Clinic,Location,c]
 L triangle2[Role,Clinic,Therapist,r]
 L triangle2[Role,Clinic,Therapist,c]
 L triangle2[Role,Clinic,Instructor,r]
 L triangle2[Role,Clinic,Instructor,c]
 L triangle2[Role,Location,Therapist,r]
 L triangle2[Role,Location,Therapist,c]
 L triangle2[Role,Location,Instructor,r]
 L triangle2[Role,Location,Instructor,c]
 L triangle2[Role,Therapist,Instructor,r]
 L triangle2[Role,Therapist,Instructor,c]
 L triangle2[User,CourseSession,Clinic,r]
 L triangle2[User,CourseSession,Clinic,c]
 L triangle2[User,CourseSession,Location,r]
 L triangle2[User,CourseSession,Location,c]
 L triangle2[User,CourseSession,Therapist,r]
 L triangle2[User,CourseSession,Therapist,c]
 L triangle2[User,CourseSession,Instructor,r]
 L triangle2[User,CourseSession,Instructor,c]
 L triangle2[User,Clinic,Location,r]
 L triangle2[User,Clinic,Location,c]
 L triangle2[User,Clinic,Therapist,r]
 L triangle2[User,Clinic,Therapist,c]
 L triangle2[User,Clinic,Instructor,r]
 L triangle2[User,Clinic,Instructor,c]
 L triangle2[User,Location,Therapist,r]
 L triangle2[User,Location,Therapist,c]
 L triangle2[User,Location,Instructor,r]
 L triangle2[User,Location,Instructor,c]
 L triangle2[User,Therapist,Instructor,r]
 L triangle2[User,Therapist,Instructor,c]
 L triangle2[CourseSession,Clinic,Location,r]
 L triangle2[CourseSession,Clinic,Location,c]
 L triangle2[CourseSession,Clinic,Therapist,r]
 L triangle2[CourseSession,Clinic,Therapist,c]
 L triangle2[CourseSession,Clinic,Instructor,r]
 L triangle2[CourseSession,Clinic,Instructor,c]
 L triangle2[CourseSession,Location,Therapist,r]
 L triangle2[CourseSession,Location,Therapist,c]
 L triangle2[CourseSession,Location,Instructor,r]
 L triangle2[CourseSession,Location,Instructor,c]
 L triangle2[CourseSession,Therapist,Instructor,r]
 L triangle2[CourseSession,Therapist,Instructor,c]
 L triangle2[Clinic,Location,Therapist,r]
 L triangle2[Clinic,Location,Therapist,c]
 L triangle2[Clinic,Location,Instructor,r]
 L triangle2[Clinic,Location,Instructor,c]
 L triangle2[Clinic,Therapist,Instructor,r]
 L triangle2[Clinic,Therapist,Instructor,c]
 L triangle2[Location,Therapist,Instructor,r]
 L triangle2[Location,Therapist,Instructor,c]
COLUMNS
 M0000001 'MARKER' 'INTORG'
 x[r,Client] centerDistAxis1[r,Client] 1 centerDistAxis2[r,Client] -1
 x[r,Client] axisVdifLB[r,Client,CertificateTemplate] -1 axisVdifLB[r,Client,Course] -1
 x[r,Client] axisVdifLB[r,Client,Role] -1 axisVdifLB[r,Client,User] -1
 x[r,Client] axisVdifLB[r,Client,CourseSession] -1 axisVdifLB[r,Client,Clinic] -1
 x[r,Client] axisVdifLB[r,Client,Location] -1 axisVdifLB[r,Client,Therapist] -1
 x[r,Client] axisVdifLB[r,Client,Instructor] -1 axisVdifRT[r,Client,CertificateTemplate] 1
 x[r,Client] axisVdifRT[r,Client,Course] 1 axisVdifRT[r,Client,Role] 1
 x[r,Client] axisVdifRT[r,Client,User] 1 axisVdifRT[r,Client,CourseSession] 1
 x[r,Client] axisVdifRT[r,Client,Clinic] 1 axisVdifRT[r,Client,Location] 1
 x[r,Client] axisVdifRT[r,Client,Therapist] 1 axisVdifRT[r,Client,Instructor] 1
 x[r,Client] compDistAxis1[r,Client,CertificateTemplate] -1 compDistAxis1[r,Client,CourseSession] -1
 x[r,Client] compDistAxis1[r,Client,Instructor] -1 compDistAxis1[r,Client,Location] -1
 x[r,Client] compDistAxis1[r,Client,User] -1 compDistAxis2[r,Client,CertificateTemplate] 1
 x[r,Client] compDistAxis2[r,Client,CourseSession] 1 compDistAxis2[r,Client,Instructor] 1
 x[r,Client] compDistAxis2[r,Client,Location] 1 compDistAxis2[r,Client,User] 1
 x[r,CertificateTemplate] centerDistAxis1[r,CertificateTemplate] 1 centerDistAxis2[r,CertificateTemplate] -1
 x[r,CertificateTemplate] axisVdifLB[r,Client,CertificateTemplate] 1 axisVdifLB[r,CertificateTemplate,Course] -1
 x[r,CertificateTemplate] axisVdifLB[r,CertificateTemplate,Role] -1 axisVdifLB[r,CertificateTemplate,User] -1
 x[r,CertificateTemplate] axisVdifLB[r,CertificateTemplate,CourseSession] -1 axisVdifLB[r,CertificateTemplate,Clinic] -1
 x[r,CertificateTemplate] axisVdifLB[r,CertificateTemplate,Location] -1 axisVdifLB[r,CertificateTemplate,Therapist] -1
 x[r,CertificateTemplate] axisVdifLB[r,CertificateTemplate,Instructor] -1 axisVdifRT[r,Client,CertificateTemplate] -1
 x[r,CertificateTemplate] axisVdifRT[r,CertificateTemplate,Course] 1 axisVdifRT[r,CertificateTemplate,Role] 1
 x[r,CertificateTemplate] axisVdifRT[r,CertificateTemplate,User] 1 axisVdifRT[r,CertificateTemplate,CourseSession] 1
 x[r,CertificateTemplate] axisVdifRT[r,CertificateTemplate,Clinic] 1 axisVdifRT[r,CertificateTemplate,Location] 1
 x[r,CertificateTemplate] axisVdifRT[r,CertificateTemplate,Therapist] 1 axisVdifRT[r,CertificateTemplate,Instructor] 1
 x[r,CertificateTemplate] compDistAxis1[r,Client,CertificateTemplate] 1 compDistAxis1[r,CertificateTemplate,CourseSession] -1
 x[r,CertificateTemplate] compDistAxis2[r,Client,CertificateTemplate] -1 compDistAxis2[r,CertificateTemplate,CourseSession] 1
 x[r,Course] centerDistAxis1[r,Course] 1 centerDistAxis2[r,Course] -1
 x[r,Course] axisVdifLB[r,Client,Course] 1 axisVdifLB[r,CertificateTemplate,Course] 1
 x[r,Course] axisVdifLB[r,Course,Role] -1 axisVdifLB[r,Course,User] -1
 x[r,Course] axisVdifLB[r,Course,CourseSession] -1 axisVdifLB[r,Course,Clinic] -1
 x[r,Course] axisVdifLB[r,Course,Location] -1 axisVdifLB[r,Course,Therapist] -1
 x[r,Course] axisVdifLB[r,Course,Instructor] -1 axisVdifRT[r,Client,Course] -1
 x[r,Course] axisVdifRT[r,CertificateTemplate,Course] -1 axisVdifRT[r,Course,Role] 1
 x[r,Course] axisVdifRT[r,Course,User] 1 axisVdifRT[r,Course,CourseSession] 1
 x[r,Course] axisVdifRT[r,Course,Clinic] 1 axisVdifRT[r,Course,Location] 1
 x[r,Course] axisVdifRT[r,Course,Therapist] 1 axisVdifRT[r,Course,Instructor] 1
 x[r,Course] compDistAxis1[r,Course,CourseSession] -1 compDistAxis2[r,Course,CourseSession] 1
 x[r,Role] centerDistAxis1[r,Role] 1 centerDistAxis2[r,Role] -1
 x[r,Role] axisVdifLB[r,Client,Role] 1 axisVdifLB[r,CertificateTemplate,Role] 1
 x[r,Role] axisVdifLB[r,Course,Role] 1 axisVdifLB[r,Role,User] -1
 x[r,Role] axisVdifLB[r,Role,CourseSession] -1 axisVdifLB[r,Role,Clinic] -1
 x[r,Role] axisVdifLB[r,Role,Location] -1 axisVdifLB[r,Role,Therapist] -1
 x[r,Role] axisVdifLB[r,Role,Instructor] -1 axisVdifRT[r,Client,Role] -1
 x[r,Role] axisVdifRT[r,CertificateTemplate,Role] -1 axisVdifRT[r,Course,Role] -1
 x[r,Role] axisVdifRT[r,Role,User] 1 axisVdifRT[r,Role,CourseSession] 1
 x[r,Role] axisVdifRT[r,Role,Clinic] 1 axisVdifRT[r,Role,Location] 1
 x[r,Role] axisVdifRT[r,Role,Therapist] 1 axisVdifRT[r,Role,Instructor] 1
 x[r,Role] compDistAxis1[r,Role,User] -1 compDistAxis2[r,Role,User] 1
 x[r,User] centerDistAxis1[r,User] 1 centerDistAxis2[r,User] -1
 x[r,User] axisVdifLB[r,Client,User] 1 axisVdifLB[r,CertificateTemplate,User] 1
 x[r,User] axisVdifLB[r,Course,User] 1 axisVdifLB[r,Role,User] 1
 x[r,User] axisVdifLB[r,User,CourseSession] -1 axisVdifLB[r,User,Clinic] -1
 x[r,User] axisVdifLB[r,User,Location] -1 axisVdifLB[r,User,Therapist] -1
 x[r,User] axisVdifLB[r,User,Instructor] -1 axisVdifRT[r,Client,User] -1
 x[r,User] axisVdifRT[r,CertificateTemplate,User] -1 axisVdifRT[r,Course,User] -1
 x[r,User] axisVdifRT[r,Role,User] -1 axisVdifRT[r,User,CourseSession] 1
 x[r,User] axisVdifRT[r,User,Clinic] 1 axisVdifRT[r,User,Location] 1
 x[r,User] axisVdifRT[r,User,Therapist] 1 axisVdifRT[r,User,Instructor] 1
 x[r,User] compDistAxis1[r,Client,User] 1 compDistAxis1[r,Role,User] 1
 x[r,User] compDistAxis1[r,User,Instructor] -1 compDistAxis1[r,User,Therapist] -1
 x[r,User] compDistAxis2[r,Client,User] -1 compDistAxis2[r,Role,User] -1
 x[r,User] compDistAxis2[r,User,Instructor] 1 compDistAxis2[r,User,Therapist] 1
 x[r,CourseSession] centerDistAxis1[r,CourseSession] 1 centerDistAxis2[r,CourseSession] -1
 x[r,CourseSession] axisVdifLB[r,Client,CourseSession] 1 axisVdifLB[r,CertificateTemplate,CourseSession] 1
 x[r,CourseSession] axisVdifLB[r,Course,CourseSession] 1 axisVdifLB[r,Role,CourseSession] 1
 x[r,CourseSession] axisVdifLB[r,User,CourseSession] 1 axisVdifLB[r,CourseSession,Clinic] -1
 x[r,CourseSession] axisVdifLB[r,CourseSession,Location] -1 axisVdifLB[r,CourseSession,Therapist] -1
 x[r,CourseSession] axisVdifLB[r,CourseSession,Instructor] -1 axisVdifRT[r,Client,CourseSession] -1
 x[r,CourseSession] axisVdifRT[r,CertificateTemplate,CourseSession] -1 axisVdifRT[r,Course,CourseSession] -1
 x[r,CourseSession] axisVdifRT[r,Role,CourseSession] -1 axisVdifRT[r,User,CourseSession] -1
 x[r,CourseSession] axisVdifRT[r,CourseSession,Clinic] 1 axisVdifRT[r,CourseSession,Location] 1
 x[r,CourseSession] axisVdifRT[r,CourseSession,Therapist] 1 axisVdifRT[r,CourseSession,Instructor] 1
 x[r,CourseSession] compDistAxis1[r,Client,CourseSession] 1 compDistAxis1[r,CertificateTemplate,CourseSession] 1
 x[r,CourseSession] compDistAxis1[r,Course,CourseSession] 1 compDistAxis1[r,CourseSession,Instructor] -1
 x[r,CourseSession] compDistAxis1[r,Location,CourseSession] 1 compDistAxis1[r,Therapist,CourseSession] 1
 x[r,CourseSession] compDistAxis2[r,Client,CourseSession] -1 compDistAxis2[r,CertificateTemplate,CourseSession] -1
 x[r,CourseSession] compDistAxis2[r,Course,CourseSession] -1 compDistAxis2[r,CourseSession,Instructor] 1
 x[r,CourseSession] compDistAxis2[r,Location,CourseSession] -1 compDistAxis2[r,Therapist,CourseSession] -1
 x[r,Clinic] centerDistAxis1[r,Clinic] 1 centerDistAxis2[r,Clinic] -1
 x[r,Clinic] axisVdifLB[r,Client,Clinic] 1 axisVdifLB[r,CertificateTemplate,Clinic] 1
 x[r,Clinic] axisVdifLB[r,Course,Clinic] 1 axisVdifLB[r,Role,Clinic] 1
 x[r,Clinic] axisVdifLB[r,User,Clinic] 1 axisVdifLB[r,CourseSession,Clinic] 1
 x[r,Clinic] axisVdifLB[r,Clinic,Location] -1 axisVdifLB[r,Clinic,Therapist] -1
 x[r,Clinic] axisVdifLB[r,Clinic,Instructor] -1 axisVdifRT[r,Client,Clinic] -1
 x[r,Clinic] axisVdifRT[r,CertificateTemplate,Clinic] -1 axisVdifRT[r,Course,Clinic] -1
 x[r,Clinic] axisVdifRT[r,Role,Clinic] -1 axisVdifRT[r,User,Clinic] -1
 x[r,Clinic] axisVdifRT[r,CourseSession,Clinic] -1 axisVdifRT[r,Clinic,Location] 1
 x[r,Clinic] axisVdifRT[r,Clinic,Therapist] 1 axisVdifRT[r,Clinic,Instructor] 1
 x[r,Clinic] compDistAxis1[r,Clinic,Location] -1 compDistAxis1[r,Clinic,Therapist] -1
 x[r,Clinic] compDistAxis2[r,Clinic,Location] 1 compDistAxis2[r,Clinic,Therapist] 1
 x[r,Location] centerDistAxis1[r,Location] 1 centerDistAxis2[r,Location] -1
 x[r,Location] axisVdifLB[r,Client,Location] 1 axisVdifLB[r,CertificateTemplate,Location] 1
 x[r,Location] axisVdifLB[r,Course,Location] 1 axisVdifLB[r,Role,Location] 1
 x[r,Location] axisVdifLB[r,User,Location] 1 axisVdifLB[r,CourseSession,Location] 1
 x[r,Location] axisVdifLB[r,Clinic,Location] 1 axisVdifLB[r,Location,Therapist] -1
 x[r,Location] axisVdifLB[r,Location,Instructor] -1 axisVdifRT[r,Client,Location] -1
 x[r,Location] axisVdifRT[r,CertificateTemplate,Location] -1 axisVdifRT[r,Course,Location] -1
 x[r,Location] axisVdifRT[r,Role,Location] -1 axisVdifRT[r,User,Location] -1
 x[r,Location] axisVdifRT[r,CourseSession,Location] -1 axisVdifRT[r,Clinic,Location] -1
 x[r,Location] axisVdifRT[r,Location,Therapist] 1 axisVdifRT[r,Location,Instructor] 1
 x[r,Location] compDistAxis1[r,Client,Location] 1 compDistAxis1[r,Instructor,Location] 1
 x[r,Location] compDistAxis1[r,Clinic,Location] 1 compDistAxis1[r,Location,Therapist] -1
 x[r,Location] compDistAxis1[r,Location,CourseSession] -1 compDistAxis2[r,Client,Location] -1
 x[r,Location] compDistAxis2[r,Instructor,Location] -1 compDistAxis2[r,Clinic,Location] -1
 x[r,Location] compDistAxis2[r,Location,Therapist] 1 compDistAxis2[r,Location,CourseSession] 1
 x[r,Therapist] centerDistAxis1[r,Therapist] 1 centerDistAxis2[r,Therapist] -1
 x[r,Therapist] axisVdifLB[r,Client,Therapist] 1 axisVdifLB[r,CertificateTemplate,Therapist] 1
 x[r,Therapist] axisVdifLB[r,Course,Therapist] 1 axisVdifLB[r,Role,Therapist] 1
 x[r,Therapist] axisVdifLB[r,User,Therapist] 1 axisVdifLB[r,CourseSession,Therapist] 1
 x[r,Therapist] axisVdifLB[r,Clinic,Therapist] 1 axisVdifLB[r,Location,Therapist] 1
 x[r,Therapist] axisVdifLB[r,Therapist,Instructor] -1 axisVdifRT[r,Client,Therapist] -1
 x[r,Therapist] axisVdifRT[r,CertificateTemplate,Therapist] -1 axisVdifRT[r,Course,Therapist] -1
 x[r,Therapist] axisVdifRT[r,Role,Therapist] -1 axisVdifRT[r,User,Therapist] -1
 x[r,Therapist] axisVdifRT[r,CourseSession,Therapist] -1 axisVdifRT[r,Clinic,Therapist] -1
 x[r,Therapist] axisVdifRT[r,Location,Therapist] -1 axisVdifRT[r,Therapist,Instructor] 1
 x[r,Therapist] compDistAxis1[r,Clinic,Therapist] 1 compDistAxis1[r,Location,Therapist] 1
 x[r,Therapist] compDistAxis1[r,Therapist,CourseSession] -1 compDistAxis1[r,User,Therapist] 1
 x[r,Therapist] compDistAxis2[r,Clinic,Therapist] -1 compDistAxis2[r,Location,Therapist] -1
 x[r,Therapist] compDistAxis2[r,Therapist,CourseSession] 1 compDistAxis2[r,User,Therapist] -1
 x[r,Instructor] centerDistAxis1[r,Instructor] 1 centerDistAxis2[r,Instructor] -1
 x[r,Instructor] axisVdifLB[r,Client,Instructor] 1 axisVdifLB[r,CertificateTemplate,Instructor] 1
 x[r,Instructor] axisVdifLB[r,Course,Instructor] 1 axisVdifLB[r,Role,Instructor] 1
 x[r,Instructor] axisVdifLB[r,User,Instructor] 1 axisVdifLB[r,CourseSession,Instructor] 1
 x[r,Instructor] axisVdifLB[r,Clinic,Instructor] 1 axisVdifLB[r,Location,Instructor] 1
 x[r,Instructor] axisVdifLB[r,Therapist,Instructor] 1 axisVdifRT[r,Client,Instructor] -1
 x[r,Instructor] axisVdifRT[r,CertificateTemplate,Instructor] -1 axisVdifRT[r,Course,Instructor] -1
 x[r,Instructor] axisVdifRT[r,Role,Instructor] -1 axisVdifRT[r,User,Instructor] -1
 x[r,Instructor] axisVdifRT[r,CourseSession,Instructor] -1 axisVdifRT[r,Clinic,Instructor] -1
 x[r,Instructor] axisVdifRT[r,Location,Instructor] -1 axisVdifRT[r,Therapist,Instructor] -1
 x[r,Instructor] compDistAxis1[r,Client,Instructor] 1 compDistAxis1[r,CourseSession,Instructor] 1
 x[r,Instructor] compDistAxis1[r,Instructor,Location] -1 compDistAxis1[r,User,Instructor] 1
 x[r,Instructor] compDistAxis2[r,Client,Instructor] -1 compDistAxis2[r,CourseSession,Instructor] -1
 x[r,Instructor] compDistAxis2[r,Instructor,Location] 1 compDistAxis2[r,User,Instructor] -1
 x[c,Client] centerDistAxis1[c,Client] 1 centerDistAxis2[c,Client] -1
 x[c,Client] axisVdifLB[c,Client,CertificateTemplate] -1 axisVdifLB[c,Client,Course] -1
 x[c,Client] axisVdifLB[c,Client,Role] -1 axisVdifLB[c,Client,User] -1
 x[c,Client] axisVdifLB[c,Client,CourseSession] -1 axisVdifLB[c,Client,Clinic] -1
 x[c,Client] axisVdifLB[c,Client,Location] -1 axisVdifLB[c,Client,Therapist] -1
 x[c,Client] axisVdifLB[c,Client,Instructor] -1 axisVdifRT[c,Client,CertificateTemplate] 1
 x[c,Client] axisVdifRT[c,Client,Course] 1 axisVdifRT[c,Client,Role] 1
 x[c,Client] axisVdifRT[c,Client,User] 1 axisVdifRT[c,Client,CourseSession] 1
 x[c,Client] axisVdifRT[c,Client,Clinic] 1 axisVdifRT[c,Client,Location] 1
 x[c,Client] axisVdifRT[c,Client,Therapist] 1 axisVdifRT[c,Client,Instructor] 1
 x[c,Client] compDistAxis1[c,Client,CertificateTemplate] -1 compDistAxis1[c,Client,CourseSession] -1
 x[c,Client] compDistAxis1[c,Client,Instructor] -1 compDistAxis1[c,Client,Location] -1
 x[c,Client] compDistAxis1[c,Client,User] -1 compDistAxis2[c,Client,CertificateTemplate] 1
 x[c,Client] compDistAxis2[c,Client,CourseSession] 1 compDistAxis2[c,Client,Instructor] 1
 x[c,Client] compDistAxis2[c,Client,Location] 1 compDistAxis2[c,Client,User] 1
 x[c,CertificateTemplate] centerDistAxis1[c,CertificateTemplate] 1 centerDistAxis2[c,CertificateTemplate] -1
 x[c,CertificateTemplate] axisVdifLB[c,Client,CertificateTemplate] 1 axisVdifLB[c,CertificateTemplate,Course] -1
 x[c,CertificateTemplate] axisVdifLB[c,CertificateTemplate,Role] -1 axisVdifLB[c,CertificateTemplate,User] -1
 x[c,CertificateTemplate] axisVdifLB[c,CertificateTemplate,CourseSession] -1 axisVdifLB[c,CertificateTemplate,Clinic] -1
 x[c,CertificateTemplate] axisVdifLB[c,CertificateTemplate,Location] -1 axisVdifLB[c,CertificateTemplate,Therapist] -1
 x[c,CertificateTemplate] axisVdifLB[c,CertificateTemplate,Instructor] -1 axisVdifRT[c,Client,CertificateTemplate] -1
 x[c,CertificateTemplate] axisVdifRT[c,CertificateTemplate,Course] 1 axisVdifRT[c,CertificateTemplate,Role] 1
 x[c,CertificateTemplate] axisVdifRT[c,CertificateTemplate,User] 1 axisVdifRT[c,CertificateTemplate,CourseSession] 1
 x[c,CertificateTemplate] axisVdifRT[c,CertificateTemplate,Clinic] 1 axisVdifRT[c,CertificateTemplate,Location] 1
 x[c,CertificateTemplate] axisVdifRT[c,CertificateTemplate,Therapist] 1 axisVdifRT[c,CertificateTemplate,Instructor] 1
 x[c,CertificateTemplate] compDistAxis1[c,Client,CertificateTemplate] 1 compDistAxis1[c,CertificateTemplate,CourseSession] -1
 x[c,CertificateTemplate] compDistAxis2[c,Client,CertificateTemplate] -1 compDistAxis2[c,CertificateTemplate,CourseSession] 1
 x[c,Course] centerDistAxis1[c,Course] 1 centerDistAxis2[c,Course] -1
 x[c,Course] axisVdifLB[c,Client,Course] 1 axisVdifLB[c,CertificateTemplate,Course] 1
 x[c,Course] axisVdifLB[c,Course,Role] -1 axisVdifLB[c,Course,User] -1
 x[c,Course] axisVdifLB[c,Course,CourseSession] -1 axisVdifLB[c,Course,Clinic] -1
 x[c,Course] axisVdifLB[c,Course,Location] -1 axisVdifLB[c,Course,Therapist] -1
 x[c,Course] axisVdifLB[c,Course,Instructor] -1 axisVdifRT[c,Client,Course] -1
 x[c,Course] axisVdifRT[c,CertificateTemplate,Course] -1 axisVdifRT[c,Course,Role] 1
 x[c,Course] axisVdifRT[c,Course,User] 1 axisVdifRT[c,Course,CourseSession] 1
 x[c,Course] axisVdifRT[c,Course,Clinic] 1 axisVdifRT[c,Course,Location] 1
 x[c,Course] axisVdifRT[c,Course,Therapist] 1 axisVdifRT[c,Course,Instructor] 1
 x[c,Course] compDistAxis1[c,Course,CourseSession] -1 compDistAxis2[c,Course,CourseSession] 1
 x[c,Role] centerDistAxis1[c,Role] 1 centerDistAxis2[c,Role] -1
 x[c,Role] axisVdifLB[c,Client,Role] 1 axisVdifLB[c,CertificateTemplate,Role] 1
 x[c,Role] axisVdifLB[c,Course,Role] 1 axisVdifLB[c,Role,User] -1
 x[c,Role] axisVdifLB[c,Role,CourseSession] -1 axisVdifLB[c,Role,Clinic] -1
 x[c,Role] axisVdifLB[c,Role,Location] -1 axisVdifLB[c,Role,Therapist] -1
 x[c,Role] axisVdifLB[c,Role,Instructor] -1 axisVdifRT[c,Client,Role] -1
 x[c,Role] axisVdifRT[c,CertificateTemplate,Role] -1 axisVdifRT[c,Course,Role] -1
 x[c,Role] axisVdifRT[c,Role,User] 1 axisVdifRT[c,Role,CourseSession] 1
 x[c,Role] axisVdifRT[c,Role,Clinic] 1 axisVdifRT[c,Role,Location] 1
 x[c,Role] axisVdifRT[c,Role,Therapist] 1 axisVdifRT[c,Role,Instructor] 1
 x[c,Role] compDistAxis1[c,Role,User] -1 compDistAxis2[c,Role,User] 1
 x[c,User] centerDistAxis1[c,User] 1 centerDistAxis2[c,User] -1
 x[c,User] axisVdifLB[c,Client,User] 1 axisVdifLB[c,CertificateTemplate,User] 1
 x[c,User] axisVdifLB[c,Course,User] 1 axisVdifLB[c,Role,User] 1
 x[c,User] axisVdifLB[c,User,CourseSession] -1 axisVdifLB[c,User,Clinic] -1
 x[c,User] axisVdifLB[c,User,Location] -1 axisVdifLB[c,User,Therapist] -1
 x[c,User] axisVdifLB[c,User,Instructor] -1 axisVdifRT[c,Client,User] -1
 x[c,User] axisVdifRT[c,CertificateTemplate,User] -1 axisVdifRT[c,Course,User] -1
 x[c,User] axisVdifRT[c,Role,User] -1 axisVdifRT[c,User,CourseSession] 1
 x[c,User] axisVdifRT[c,User,Clinic] 1 axisVdifRT[c,User,Location] 1
 x[c,User] axisVdifRT[c,User,Therapist] 1 axisVdifRT[c,User,Instructor] 1
 x[c,User] compDistAxis1[c,Client,User] 1 compDistAxis1[c,Role,User] 1
 x[c,User] compDistAxis1[c,User,Instructor] -1 compDistAxis1[c,User,Therapist] -1
 x[c,User] compDistAxis2[c,Client,User] -1 compDistAxis2[c,Role,User] -1
 x[c,User] compDistAxis2[c,User,Instructor] 1 compDistAxis2[c,User,Therapist] 1
 x[c,CourseSession] centerDistAxis1[c,CourseSession] 1 centerDistAxis2[c,CourseSession] -1
 x[c,CourseSession] axisVdifLB[c,Client,CourseSession] 1 axisVdifLB[c,CertificateTemplate,CourseSession] 1
 x[c,CourseSession] axisVdifLB[c,Course,CourseSession] 1 axisVdifLB[c,Role,CourseSession] 1
 x[c,CourseSession] axisVdifLB[c,User,CourseSession] 1 axisVdifLB[c,CourseSession,Clinic] -1
 x[c,CourseSession] axisVdifLB[c,CourseSession,Location] -1 axisVdifLB[c,CourseSession,Therapist] -1
 x[c,CourseSession] axisVdifLB[c,CourseSession,Instructor] -1 axisVdifRT[c,Client,CourseSession] -1
 x[c,CourseSession] axisVdifRT[c,CertificateTemplate,CourseSession] -1 axisVdifRT[c,Course,CourseSession] -1
 x[c,CourseSession] axisVdifRT[c,Role,CourseSession] -1 axisVdifRT[c,User,CourseSession] -1
 x[c,CourseSession] axisVdifRT[c,CourseSession,Clinic] 1 axisVdifRT[c,CourseSession,Location] 1
 x[c,CourseSession] axisVdifRT[c,CourseSession,Therapist] 1 axisVdifRT[c,CourseSession,Instructor] 1
 x[c,CourseSession] compDistAxis1[c,Client,CourseSession] 1 compDistAxis1[c,CertificateTemplate,CourseSession] 1
 x[c,CourseSession] compDistAxis1[c,Course,CourseSession] 1 compDistAxis1[c,CourseSession,Instructor] -1
 x[c,CourseSession] compDistAxis1[c,Location,CourseSession] 1 compDistAxis1[c,Therapist,CourseSession] 1
 x[c,CourseSession] compDistAxis2[c,Client,CourseSession] -1 compDistAxis2[c,CertificateTemplate,CourseSession] -1
 x[c,CourseSession] compDistAxis2[c,Course,CourseSession] -1 compDistAxis2[c,CourseSession,Instructor] 1
 x[c,CourseSession] compDistAxis2[c,Location,CourseSession] -1 compDistAxis2[c,Therapist,CourseSession] -1
 x[c,Clinic] centerDistAxis1[c,Clinic] 1 centerDistAxis2[c,Clinic] -1
 x[c,Clinic] axisVdifLB[c,Client,Clinic] 1 axisVdifLB[c,CertificateTemplate,Clinic] 1
 x[c,Clinic] axisVdifLB[c,Course,Clinic] 1 axisVdifLB[c,Role,Clinic] 1
 x[c,Clinic] axisVdifLB[c,User,Clinic] 1 axisVdifLB[c,CourseSession,Clinic] 1
 x[c,Clinic] axisVdifLB[c,Clinic,Location] -1 axisVdifLB[c,Clinic,Therapist] -1
 x[c,Clinic] axisVdifLB[c,Clinic,Instructor] -1 axisVdifRT[c,Client,Clinic] -1
 x[c,Clinic] axisVdifRT[c,CertificateTemplate,Clinic] -1 axisVdifRT[c,Course,Clinic] -1
 x[c,Clinic] axisVdifRT[c,Role,Clinic] -1 axisVdifRT[c,User,Clinic] -1
 x[c,Clinic] axisVdifRT[c,CourseSession,Clinic] -1 axisVdifRT[c,Clinic,Location] 1
 x[c,Clinic] axisVdifRT[c,Clinic,Therapist] 1 axisVdifRT[c,Clinic,Instructor] 1
 x[c,Clinic] compDistAxis1[c,Clinic,Location] -1 compDistAxis1[c,Clinic,Therapist] -1
 x[c,Clinic] compDistAxis2[c,Clinic,Location] 1 compDistAxis2[c,Clinic,Therapist] 1
 x[c,Location] centerDistAxis1[c,Location] 1 centerDistAxis2[c,Location] -1
 x[c,Location] axisVdifLB[c,Client,Location] 1 axisVdifLB[c,CertificateTemplate,Location] 1
 x[c,Location] axisVdifLB[c,Course,Location] 1 axisVdifLB[c,Role,Location] 1
 x[c,Location] axisVdifLB[c,User,Location] 1 axisVdifLB[c,CourseSession,Location] 1
 x[c,Location] axisVdifLB[c,Clinic,Location] 1 axisVdifLB[c,Location,Therapist] -1
 x[c,Location] axisVdifLB[c,Location,Instructor] -1 axisVdifRT[c,Client,Location] -1
 x[c,Location] axisVdifRT[c,CertificateTemplate,Location] -1 axisVdifRT[c,Course,Location] -1
 x[c,Location] axisVdifRT[c,Role,Location] -1 axisVdifRT[c,User,Location] -1
 x[c,Location] axisVdifRT[c,CourseSession,Location] -1 axisVdifRT[c,Clinic,Location] -1
 x[c,Location] axisVdifRT[c,Location,Therapist] 1 axisVdifRT[c,Location,Instructor] 1
 x[c,Location] compDistAxis1[c,Client,Location] 1 compDistAxis1[c,Instructor,Location] 1
 x[c,Location] compDistAxis1[c,Clinic,Location] 1 compDistAxis1[c,Location,Therapist] -1
 x[c,Location] compDistAxis1[c,Location,CourseSession] -1 compDistAxis2[c,Client,Location] -1
 x[c,Location] compDistAxis2[c,Instructor,Location] -1 compDistAxis2[c,Clinic,Location] -1
 x[c,Location] compDistAxis2[c,Location,Therapist] 1 compDistAxis2[c,Location,CourseSession] 1
 x[c,Therapist] centerDistAxis1[c,Therapist] 1 centerDistAxis2[c,Therapist] -1
 x[c,Therapist] axisVdifLB[c,Client,Therapist] 1 axisVdifLB[c,CertificateTemplate,Therapist] 1
 x[c,Therapist] axisVdifLB[c,Course,Therapist] 1 axisVdifLB[c,Role,Therapist] 1
 x[c,Therapist] axisVdifLB[c,User,Therapist] 1 axisVdifLB[c,CourseSession,Therapist] 1
 x[c,Therapist] axisVdifLB[c,Clinic,Therapist] 1 axisVdifLB[c,Location,Therapist] 1
 x[c,Therapist] axisVdifLB[c,Therapist,Instructor] -1 axisVdifRT[c,Client,Therapist] -1
 x[c,Therapist] axisVdifRT[c,CertificateTemplate,Therapist] -1 axisVdifRT[c,Course,Therapist] -1
 x[c,Therapist] axisVdifRT[c,Role,Therapist] -1 axisVdifRT[c,User,Therapist] -1
 x[c,Therapist] axisVdifRT[c,CourseSession,Therapist] -1 axisVdifRT[c,Clinic,Therapist] -1
 x[c,Therapist] axisVdifRT[c,Location,Therapist] -1 axisVdifRT[c,Therapist,Instructor] 1
 x[c,Therapist] compDistAxis1[c,Clinic,Therapist] 1 compDistAxis1[c,Location,Therapist] 1
 x[c,Therapist] compDistAxis1[c,Therapist,CourseSession] -1 compDistAxis1[c,User,Therapist] 1
 x[c,Therapist] compDistAxis2[c,Clinic,Therapist] -1 compDistAxis2[c,Location,Therapist] -1
 x[c,Therapist] compDistAxis2[c,Therapist,CourseSession] 1 compDistAxis2[c,User,Therapist] -1
 x[c,Instructor] centerDistAxis1[c,Instructor] 1 centerDistAxis2[c,Instructor] -1
 x[c,Instructor] axisVdifLB[c,Client,Instructor] 1 axisVdifLB[c,CertificateTemplate,Instructor] 1
 x[c,Instructor] axisVdifLB[c,Course,Instructor] 1 axisVdifLB[c,Role,Instructor] 1
 x[c,Instructor] axisVdifLB[c,User,Instructor] 1 axisVdifLB[c,CourseSession,Instructor] 1
 x[c,Instructor] axisVdifLB[c,Clinic,Instructor] 1 axisVdifLB[c,Location,Instructor] 1
 x[c,Instructor] axisVdifLB[c,Therapist,Instructor] 1 axisVdifRT[c,Client,Instructor] -1
 x[c,Instructor] axisVdifRT[c,CertificateTemplate,Instructor] -1 axisVdifRT[c,Course,Instructor] -1
 x[c,Instructor] axisVdifRT[c,Role,Instructor] -1 axisVdifRT[c,User,Instructor] -1
 x[c,Instructor] axisVdifRT[c,CourseSession,Instructor] -1 axisVdifRT[c,Clinic,Instructor] -1
 x[c,Instructor] axisVdifRT[c,Location,Instructor] -1 axisVdifRT[c,Therapist,Instructor] -1
 x[c,Instructor] compDistAxis1[c,Client,Instructor] 1 compDistAxis1[c,CourseSession,Instructor] 1
 x[c,Instructor] compDistAxis1[c,Instructor,Location] -1 compDistAxis1[c,User,Instructor] 1
 x[c,Instructor] compDistAxis2[c,Client,Instructor] -1 compDistAxis2[c,CourseSession,Instructor] -1
 x[c,Instructor] compDistAxis2[c,Instructor,Location] 1 compDistAxis2[c,User,Instructor] -1
 M0000002 'MARKER' 'INTEND'
 center_dist[r,Client] drawing 1 centerDistAxis1[r,Client] 1
 center_dist[r,Client] centerDistAxis2[r,Client] 1 CenterDistSumLB 1
 center_dist[r,CertificateTemplate] drawing 1 centerDistAxis1[r,CertificateTemplate] 1
 center_dist[r,CertificateTemplate] centerDistAxis2[r,CertificateTemplate] 1 CenterDistSumLB 1
 center_dist[r,Course] drawing 1 centerDistAxis1[r,Course] 1
 center_dist[r,Course] centerDistAxis2[r,Course] 1 CenterDistSumLB 1
 center_dist[r,Role] drawing 1 centerDistAxis1[r,Role] 1
 center_dist[r,Role] centerDistAxis2[r,Role] 1 CenterDistSumLB 1
 center_dist[r,User] drawing 1 centerDistAxis1[r,User] 1
 center_dist[r,User] centerDistAxis2[r,User] 1 CenterDistSumLB 1
 center_dist[r,CourseSession] drawing 1 centerDistAxis1[r,CourseSession] 1
 center_dist[r,CourseSession] centerDistAxis2[r,CourseSession] 1 CenterDistSumLB 1
 center_dist[r,Clinic] drawing 1 centerDistAxis1[r,Clinic] 1
 center_dist[r,Clinic] centerDistAxis2[r,Clinic] 1 CenterDistSumLB 1
 center_dist[r,Location] drawing 1 centerDistAxis1[r,Location] 1
 center_dist[r,Location] centerDistAxis2[r,Location] 1 CenterDistSumLB 1
 center_dist[r,Therapist] drawing 1 centerDistAxis1[r,Therapist] 1
 center_dist[r,Therapist] centerDistAxis2[r,Therapist] 1 CenterDistSumLB 1
 center_dist[r,Instructor] drawing 1 centerDistAxis1[r,Instructor] 1
 center_dist[r,Instructor] centerDistAxis2[r,Instructor] 1 CenterDistSumLB 1
 center_dist[c,Client] drawing 1 centerDistAxis1[c,Client] 1
 center_dist[c,Client] centerDistAxis2[c,Client] 1 CenterDistSumLB 1
 center_dist[c,CertificateTemplate] drawing 1 centerDistAxis1[c,CertificateTemplate] 1
 center_dist[c,CertificateTemplate] centerDistAxis2[c,CertificateTemplate] 1 CenterDistSumLB 1
 center_dist[c,Course] drawing 1 centerDistAxis1[c,Course] 1
 center_dist[c,Course] centerDistAxis2[c,Course] 1 CenterDistSumLB 1
 center_dist[c,Role] drawing 1 centerDistAxis1[c,Role] 1
 center_dist[c,Role] centerDistAxis2[c,Role] 1 CenterDistSumLB 1
 center_dist[c,User] drawing 1 centerDistAxis1[c,User] 1
 center_dist[c,User] centerDistAxis2[c,User] 1 CenterDistSumLB 1
 center_dist[c,CourseSession] drawing 1 centerDistAxis1[c,CourseSession] 1
 center_dist[c,CourseSession] centerDistAxis2[c,CourseSession] 1 CenterDistSumLB 1
 center_dist[c,Clinic] drawing 1 centerDistAxis1[c,Clinic] 1
 center_dist[c,Clinic] centerDistAxis2[c,Clinic] 1 CenterDistSumLB 1
 center_dist[c,Location] drawing 1 centerDistAxis1[c,Location] 1
 center_dist[c,Location] centerDistAxis2[c,Location] 1 CenterDistSumLB 1
 center_dist[c,Therapist] drawing 1 centerDistAxis1[c,Therapist] 1
 center_dist[c,Therapist] centerDistAxis2[c,Therapist] 1 CenterDistSumLB 1
 center_dist[c,Instructor] drawing 1 centerDistAxis1[c,Instructor] 1
 center_dist[c,Instructor] centerDistAxis2[c,Instructor] 1 CenterDistSumLB 1
 dist[r,Client,CertificateTemplate] drawing 86 compDistAxis1[r,Client,CertificateTemplate] 1
 dist[r,Client,CertificateTemplate] compDistAxis2[r,Client,CertificateTemplate] 1 DistSumLB[Client,CertificateTemplate] 1
 dist[r,Client,CertificateTemplate] DistAxisLB1[r,Client,CertificateTemplate] 1
 dist[r,Client,CourseSession] drawing 86 compDistAxis1[r,Client,CourseSession] 1
 dist[r,Client,CourseSession] compDistAxis2[r,Client,CourseSession] 1 DistSumLB[Client,CourseSession] 1
 dist[r,Client,CourseSession] DistAxisLB1[r,Client,CourseSession] 1
 dist[r,Client,Instructor] drawing 86 compDistAxis1[r,Client,Instructor] 1
 dist[r,Client,Instructor] compDistAxis2[r,Client,Instructor] 1 DistSumLB[Client,Instructor] 1
 dist[r,Client,Instructor] DistAxisLB1[r,Client,Instructor] 1
 dist[r,Client,Location] drawing 86 compDistAxis1[r,Client,Location] 1
 dist[r,Client,Location] compDistAxis2[r,Client,Location] 1 DistSumLB[Client,Location] 1
 dist[r,Client,Location] DistAxisLB1[r,Client,Location] 1
 dist[r,Client,User] drawing 86 compDistAxis1[r,Client,User] 1
 dist[r,Client,User] compDistAxis2[r,Client,User] 1 DistSumLB[Client,User] 1
 dist[r,Client,User] DistAxisLB1[r,Client,User] 1
 dist[r,CertificateTemplate,CourseSession] drawing 86 compDistAxis1[r,CertificateTemplate,CourseSession] 1
 dist[r,CertificateTemplate,CourseSession] compDistAxis2[r,CertificateTemplate,CourseSession] 1 DistSumLB[CertificateTemplate,CourseSession] 1
 dist[r,CertificateTemplate,CourseSession] DistAxisLB1[r,CertificateTemplate,CourseSession] 1
 dist[r,Course,CourseSession] drawing 86 compDistAxis1[r,Course,CourseSession] 1
 dist[r,Course,CourseSession] compDistAxis2[r,Course,CourseSession] 1 DistSumLB[Course,CourseSession] 1
 dist[r,Course,CourseSession] DistAxisLB1[r,Course,CourseSession] 1
 dist[r,CourseSession,Instructor] drawing 86 compDistAxis1[r,CourseSession,Instructor] 1
 dist[r,CourseSession,Instructor] compDistAxis2[r,CourseSession,Instructor] 1 DistSumLB[CourseSession,Instructor] 1
 dist[r,CourseSession,Instructor] DistAxisLB1[r,CourseSession,Instructor] 1
 dist[r,Instructor,Location] drawing 86 compDistAxis1[r,Instructor,Location] 1
 dist[r,Instructor,Location] compDistAxis2[r,Instructor,Location] 1 DistSumLB[Instructor,Location] 1
 dist[r,Instructor,Location] DistAxisLB1[r,Instructor,Location] 1
 dist[r,Clinic,Location] drawing 86 compDistAxis1[r,Clinic,Location] 1
 dist[r,Clinic,Location] compDistAxis2[r,Clinic,Location] 1 DistSumLB[Clinic,Location] 1
 dist[r,Clinic,Location] DistAxisLB1[r,Clinic,Location] 1
 dist[r,Clinic,Therapist] drawing 86 compDistAxis1[r,Clinic,Therapist] 1
 dist[r,Clinic,Therapist] compDistAxis2[r,Clinic,Therapist] 1 DistSumLB[Clinic,Therapist] 1
 dist[r,Clinic,Therapist] DistAxisLB1[r,Clinic,Therapist] 1
 dist[r,Location,Therapist] drawing 86 compDistAxis1[r,Location,Therapist] 1
 dist[r,Location,Therapist] compDistAxis2[r,Location,Therapist] 1 DistSumLB[Location,Therapist] 1
 dist[r,Location,Therapist] DistAxisLB1[r,Location,Therapist] 1
 dist[r,Location,CourseSession] drawing 86 compDistAxis1[r,Location,CourseSession] 1
 dist[r,Location,CourseSession] compDistAxis2[r,Location,CourseSession] 1 DistSumLB[Location,CourseSession] 1
 dist[r,Location,CourseSession] DistAxisLB1[r,Location,CourseSession] 1
 dist[r,Therapist,CourseSession] drawing 86 compDistAxis1[r,Therapist,CourseSession] 1
 dist[r,Therapist,CourseSession] compDistAxis2[r,Therapist,CourseSession] 1 DistSumLB[Therapist,CourseSession] 1
 dist[r,Therapist,CourseSession] DistAxisLB1[r,Therapist,CourseSession] 1
 dist[r,Role,User] drawing 86 compDistAxis1[r,Role,User] 1
 dist[r,Role,User] compDistAxis2[r,Role,User] 1 DistSumLB[Role,User] 1
 dist[r,Role,User] DistAxisLB1[r,Role,User] 1
 dist[r,User,Instructor] drawing 86 compDistAxis1[r,User,Instructor] 1
 dist[r,User,Instructor] compDistAxis2[r,User,Instructor] 1 DistSumLB[User,Instructor] 1
 dist[r,User,Instructor] DistAxisLB1[r,User,Instructor] 1
 dist[r,User,Therapist] drawing 86 compDistAxis1[r,User,Therapist] 1
 dist[r,User,Therapist] compDistAxis2[r,User,Therapist] 1 DistSumLB[User,Therapist] 1
 dist[r,User,Therapist] DistAxisLB1[r,User,Therapist] 1
 dist[c,Client,CertificateTemplate] drawing 86 compDistAxis1[c,Client,CertificateTemplate] 1
 dist[c,Client,CertificateTemplate] compDistAxis2[c,Client,CertificateTemplate] 1 DistSumLB[Client,CertificateTemplate] 1
 dist[c,Client,CertificateTemplate] DistAxisLB1[c,Client,CertificateTemplate] 1
 dist[c,Client,CourseSession] drawing 86 compDistAxis1[c,Client,CourseSession] 1
 dist[c,Client,CourseSession] compDistAxis2[c,Client,CourseSession] 1 DistSumLB[Client,CourseSession] 1
 dist[c,Client,CourseSession] DistAxisLB1[c,Client,CourseSession] 1
 dist[c,Client,Instructor] drawing 86 compDistAxis1[c,Client,Instructor] 1
 dist[c,Client,Instructor] compDistAxis2[c,Client,Instructor] 1 DistSumLB[Client,Instructor] 1
 dist[c,Client,Instructor] DistAxisLB1[c,Client,Instructor] 1
 dist[c,Client,Location] drawing 86 compDistAxis1[c,Client,Location] 1
 dist[c,Client,Location] compDistAxis2[c,Client,Location] 1 DistSumLB[Client,Location] 1
 dist[c,Client,Location] DistAxisLB1[c,Client,Location] 1
 dist[c,Client,User] drawing 86 compDistAxis1[c,Client,User] 1
 dist[c,Client,User] compDistAxis2[c,Client,User] 1 DistSumLB[Client,User] 1
 dist[c,Client,User] DistAxisLB1[c,Client,User] 1
 dist[c,CertificateTemplate,CourseSession] drawing 86 compDistAxis1[c,CertificateTemplate,CourseSession] 1
 dist[c,CertificateTemplate,CourseSession] compDistAxis2[c,CertificateTemplate,CourseSession] 1 DistSumLB[CertificateTemplate,CourseSession] 1
 dist[c,CertificateTemplate,CourseSession] DistAxisLB1[c,CertificateTemplate,CourseSession] 1
 dist[c,Course,CourseSession] drawing 86 compDistAxis1[c,Course,CourseSession] 1
 dist[c,Course,CourseSession] compDistAxis2[c,Course,CourseSession] 1 DistSumLB[Course,CourseSession] 1
 dist[c,Course,CourseSession] DistAxisLB1[c,Course,CourseSession] 1
 dist[c,CourseSession,Instructor] drawing 86 compDistAxis1[c,CourseSession,Instructor] 1
 dist[c,CourseSession,Instructor] compDistAxis2[c,CourseSession,Instructor] 1 DistSumLB[CourseSession,Instructor] 1
 dist[c,CourseSession,Instructor] DistAxisLB1[c,CourseSession,Instructor] 1
 dist[c,Instructor,Location] drawing 86 compDistAxis1[c,Instructor,Location] 1
 dist[c,Instructor,Location] compDistAxis2[c,Instructor,Location] 1 DistSumLB[Instructor,Location] 1
 dist[c,Instructor,Location] DistAxisLB1[c,Instructor,Location] 1
 dist[c,Clinic,Location] drawing 86 compDistAxis1[c,Clinic,Location] 1
 dist[c,Clinic,Location] compDistAxis2[c,Clinic,Location] 1 DistSumLB[Clinic,Location] 1
 dist[c,Clinic,Location] DistAxisLB1[c,Clinic,Location] 1
 dist[c,Clinic,Therapist] drawing 86 compDistAxis1[c,Clinic,Therapist] 1
 dist[c,Clinic,Therapist] compDistAxis2[c,Clinic,Therapist] 1 DistSumLB[Clinic,Therapist] 1
 dist[c,Clinic,Therapist] DistAxisLB1[c,Clinic,Therapist] 1
 dist[c,Location,Therapist] drawing 86 compDistAxis1[c,Location,Therapist] 1
 dist[c,Location,Therapist] compDistAxis2[c,Location,Therapist] 1 DistSumLB[Location,Therapist] 1
 dist[c,Location,Therapist] DistAxisLB1[c,Location,Therapist] 1
 dist[c,Location,CourseSession] drawing 86 compDistAxis1[c,Location,CourseSession] 1
 dist[c,Location,CourseSession] compDistAxis2[c,Location,CourseSession] 1 DistSumLB[Location,CourseSession] 1
 dist[c,Location,CourseSession] DistAxisLB1[c,Location,CourseSession] 1
 dist[c,Therapist,CourseSession] drawing 86 compDistAxis1[c,Therapist,CourseSession] 1
 dist[c,Therapist,CourseSession] compDistAxis2[c,Therapist,CourseSession] 1 DistSumLB[Therapist,CourseSession] 1
 dist[c,Therapist,CourseSession] DistAxisLB1[c,Therapist,CourseSession] 1
 dist[c,Role,User] drawing 86 compDistAxis1[c,Role,User] 1
 dist[c,Role,User] compDistAxis2[c,Role,User] 1 DistSumLB[Role,User] 1
 dist[c,Role,User] DistAxisLB1[c,Role,User] 1
 dist[c,User,Instructor] drawing 86 compDistAxis1[c,User,Instructor] 1
 dist[c,User,Instructor] compDistAxis2[c,User,Instructor] 1 DistSumLB[User,Instructor] 1
 dist[c,User,Instructor] DistAxisLB1[c,User,Instructor] 1
 dist[c,User,Therapist] drawing 86 compDistAxis1[c,User,Therapist] 1
 dist[c,User,Therapist] compDistAxis2[c,User,Therapist] 1 DistSumLB[User,Therapist] 1
 dist[c,User,Therapist] DistAxisLB1[c,User,Therapist] 1
 M0000003 'MARKER' 'INTORG'
 zV[r,rt,Client,CertificateTemplate] choose1[Client,CertificateTemplate] 1 axisVdifRT[r,Client,CertificateTemplate] -49
 zV[r,rt,Client,CertificateTemplate] DistAxisLB1[c,Client,CertificateTemplate] 11.5 triangle1[Client,CertificateTemplate,Course,r] 1
 zV[r,rt,Client,CertificateTemplate] triangle1[Client,CertificateTemplate,Role,r] 1 triangle1[Client,CertificateTemplate,User,r] 1
 zV[r,rt,Client,CertificateTemplate] triangle1[Client,CertificateTemplate,CourseSession,r] 1 triangle1[Client,CertificateTemplate,Clinic,r] 1
 zV[r,rt,Client,CertificateTemplate] triangle1[Client,CertificateTemplate,Location,r] 1 triangle1[Client,CertificateTemplate,Therapist,r] 1
 zV[r,rt,Client,CertificateTemplate] triangle1[Client,CertificateTemplate,Instructor,r] 1
 zV[c,rt,Client,CertificateTemplate] choose1[Client,CertificateTemplate] 1 axisVdifRT[c,Client,CertificateTemplate] -54
 zV[c,rt,Client,CertificateTemplate] DistAxisLB1[r,Client,CertificateTemplate] 8 triangle1[Client,CertificateTemplate,Course,c] 1
 zV[c,rt,Client,CertificateTemplate] triangle1[Client,CertificateTemplate,Role,c] 1 triangle1[Client,CertificateTemplate,User,c] 1
 zV[c,rt,Client,CertificateTemplate] triangle1[Client,CertificateTemplate,CourseSession,c] 1 triangle1[Client,CertificateTemplate,Clinic,c] 1
 zV[c,rt,Client,CertificateTemplate] triangle1[Client,CertificateTemplate,Location,c] 1 triangle1[Client,CertificateTemplate,Therapist,c] 1
 zV[c,rt,Client,CertificateTemplate] triangle1[Client,CertificateTemplate,Instructor,c] 1
 zV[r,lb,Client,CertificateTemplate] choose1[Client,CertificateTemplate] 1 axisVdifLB[r,Client,CertificateTemplate] -51
 zV[r,lb,Client,CertificateTemplate] DistAxisLB1[c,Client,CertificateTemplate] 11.5 triangle2[Client,CertificateTemplate,Course,r] 1
 zV[r,lb,Client,CertificateTemplate] triangle2[Client,CertificateTemplate,Role,r] 1 triangle2[Client,CertificateTemplate,User,r] 1
 zV[r,lb,Client,CertificateTemplate] triangle2[Client,CertificateTemplate,CourseSession,r] 1 triangle2[Client,CertificateTemplate,Clinic,r] 1
 zV[r,lb,Client,CertificateTemplate] triangle2[Client,CertificateTemplate,Location,r] 1 triangle2[Client,CertificateTemplate,Therapist,r] 1
 zV[r,lb,Client,CertificateTemplate] triangle2[Client,CertificateTemplate,Instructor,r] 1
 zV[c,lb,Client,CertificateTemplate] choose1[Client,CertificateTemplate] 1 axisVdifLB[c,Client,CertificateTemplate] -53
 zV[c,lb,Client,CertificateTemplate] DistAxisLB1[r,Client,CertificateTemplate] 8 triangle2[Client,CertificateTemplate,Course,c] 1
 zV[c,lb,Client,CertificateTemplate] triangle2[Client,CertificateTemplate,Role,c] 1 triangle2[Client,CertificateTemplate,User,c] 1
 zV[c,lb,Client,CertificateTemplate] triangle2[Client,CertificateTemplate,CourseSession,c] 1 triangle2[Client,CertificateTemplate,Clinic,c] 1
 zV[c,lb,Client,CertificateTemplate] triangle2[Client,CertificateTemplate,Location,c] 1 triangle2[Client,CertificateTemplate,Therapist,c] 1
 zV[c,lb,Client,CertificateTemplate] triangle2[Client,CertificateTemplate,Instructor,c] 1
 zV[r,rt,Client,Course] choose1[Client,Course] 1 axisVdifRT[r,Client,Course] -48
 zV[r,rt,Client,Course] triangle1[Client,Course,Role,r] 1 triangle1[Client,Course,User,r] 1
 zV[r,rt,Client,Course] triangle1[Client,Course,CourseSession,r] 1 triangle1[Client,Course,Clinic,r] 1
 zV[r,rt,Client,Course] triangle1[Client,Course,Location,r] 1 triangle1[Client,Course,Therapist,r] 1
 zV[r,rt,Client,Course] triangle1[Client,Course,Instructor,r] 1 triangle2[Client,CertificateTemplate,Course,r] 1
 zV[c,rt,Client,Course] choose1[Client,Course] 1 axisVdifRT[c,Client,Course] -52
 zV[c,rt,Client,Course] triangle1[Client,Course,Role,c] 1 triangle1[Client,Course,User,c] 1
 zV[c,rt,Client,Course] triangle1[Client,Course,CourseSession,c] 1 triangle1[Client,Course,Clinic,c] 1
 zV[c,rt,Client,Course] triangle1[Client,Course,Location,c] 1 triangle1[Client,Course,Therapist,c] 1
 zV[c,rt,Client,Course] triangle1[Client,Course,Instructor,c] 1 triangle2[Client,CertificateTemplate,Course,c] 1
 zV[r,lb,Client,Course] choose1[Client,Course] 1 axisVdifLB[r,Client,Course] -51
 zV[r,lb,Client,Course] triangle1[Client,CertificateTemplate,Course,r] 1 triangle2[Client,Course,Role,r] 1
 zV[r,lb,Client,Course] triangle2[Client,Course,User,r] 1 triangle2[Client,Course,CourseSession,r] 1
 zV[r,lb,Client,Course] triangle2[Client,Course,Clinic,r] 1 triangle2[Client,Course,Location,r] 1
 zV[r,lb,Client,Course] triangle2[Client,Course,Therapist,r] 1 triangle2[Client,Course,Instructor,r] 1
 zV[c,lb,Client,Course] choose1[Client,Course] 1 axisVdifLB[c,Client,Course] -53
 zV[c,lb,Client,Course] triangle1[Client,CertificateTemplate,Course,c] 1 triangle2[Client,Course,Role,c] 1
 zV[c,lb,Client,Course] triangle2[Client,Course,User,c] 1 triangle2[Client,Course,CourseSession,c] 1
 zV[c,lb,Client,Course] triangle2[Client,Course,Clinic,c] 1 triangle2[Client,Course,Location,c] 1
 zV[c,lb,Client,Course] triangle2[Client,Course,Therapist,c] 1 triangle2[Client,Course,Instructor,c] 1
 zV[r,rt,Client,Role] choose1[Client,Role] 1 axisVdifRT[r,Client,Role] -48
 zV[r,rt,Client,Role] triangle1[Client,Role,User,r] 1 triangle1[Client,Role,CourseSession,r] 1
 zV[r,rt,Client,Role] triangle1[Client,Role,Clinic,r] 1 triangle1[Client,Role,Location,r] 1
 zV[r,rt,Client,Role] triangle1[Client,Role,Therapist,r] 1 triangle1[Client,Role,Instructor,r] 1
 zV[r,rt,Client,Role] triangle2[Client,CertificateTemplate,Role,r] 1 triangle2[Client,Course,Role,r] 1
 zV[c,rt,Client,Role] choose1[Client,Role] 1 axisVdifRT[c,Client,Role] -52
 zV[c,rt,Client,Role] triangle1[Client,Role,User,c] 1 triangle1[Client,Role,CourseSession,c] 1
 zV[c,rt,Client,Role] triangle1[Client,Role,Clinic,c] 1 triangle1[Client,Role,Location,c] 1
 zV[c,rt,Client,Role] triangle1[Client,Role,Therapist,c] 1 triangle1[Client,Role,Instructor,c] 1
 zV[c,rt,Client,Role] triangle2[Client,CertificateTemplate,Role,c] 1 triangle2[Client,Course,Role,c] 1
 zV[r,lb,Client,Role] choose1[Client,Role] 1 axisVdifLB[r,Client,Role] -51
 zV[r,lb,Client,Role] triangle1[Client,CertificateTemplate,Role,r] 1 triangle1[Client,Course,Role,r] 1
 zV[r,lb,Client,Role] triangle2[Client,Role,User,r] 1 triangle2[Client,Role,CourseSession,r] 1
 zV[r,lb,Client,Role] triangle2[Client,Role,Clinic,r] 1 triangle2[Client,Role,Location,r] 1
 zV[r,lb,Client,Role] triangle2[Client,Role,Therapist,r] 1 triangle2[Client,Role,Instructor,r] 1
 zV[c,lb,Client,Role] choose1[Client,Role] 1 axisVdifLB[c,Client,Role] -53
 zV[c,lb,Client,Role] triangle1[Client,CertificateTemplate,Role,c] 1 triangle1[Client,Course,Role,c] 1
 zV[c,lb,Client,Role] triangle2[Client,Role,User,c] 1 triangle2[Client,Role,CourseSession,c] 1
 zV[c,lb,Client,Role] triangle2[Client,Role,Clinic,c] 1 triangle2[Client,Role,Location,c] 1
 zV[c,lb,Client,Role] triangle2[Client,Role,Therapist,c] 1 triangle2[Client,Role,Instructor,c] 1
 zV[r,rt,Client,User] choose1[Client,User] 1 axisVdifRT[r,Client,User] -49
 zV[r,rt,Client,User] DistAxisLB1[c,Client,User] 11 triangle1[Client,User,CourseSession,r] 1
 zV[r,rt,Client,User] triangle1[Client,User,Clinic,r] 1 triangle1[Client,User,Location,r] 1
 zV[r,rt,Client,User] triangle1[Client,User,Therapist,r] 1 triangle1[Client,User,Instructor,r] 1
 zV[r,rt,Client,User] triangle2[Client,CertificateTemplate,User,r] 1 triangle2[Client,Course,User,r] 1
 zV[r,rt,Client,User] triangle2[Client,Role,User,r] 1
 zV[c,rt,Client,User] choose1[Client,User] 1 axisVdifRT[c,Client,User] -53
 zV[c,rt,Client,User] DistAxisLB1[r,Client,User] 8 triangle1[Client,User,CourseSession,c] 1
 zV[c,rt,Client,User] triangle1[Client,User,Clinic,c] 1 triangle1[Client,User,Location,c] 1
 zV[c,rt,Client,User] triangle1[Client,User,Therapist,c] 1 triangle1[Client,User,Instructor,c] 1
 zV[c,rt,Client,User] triangle2[Client,CertificateTemplate,User,c] 1 triangle2[Client,Course,User,c] 1
 zV[c,rt,Client,User] triangle2[Client,Role,User,c] 1
 zV[r,lb,Client,User] choose1[Client,User] 1 axisVdifLB[r,Client,User] -51
 zV[r,lb,Client,User] DistAxisLB1[c,Client,User] 11 triangle1[Client,CertificateTemplate,User,r] 1
 zV[r,lb,Client,User] triangle1[Client,Course,User,r] 1 triangle1[Client,Role,User,r] 1
 zV[r,lb,Client,User] triangle2[Client,User,CourseSession,r] 1 triangle2[Client,User,Clinic,r] 1
 zV[r,lb,Client,User] triangle2[Client,User,Location,r] 1 triangle2[Client,User,Therapist,r] 1
 zV[r,lb,Client,User] triangle2[Client,User,Instructor,r] 1
 zV[c,lb,Client,User] choose1[Client,User] 1 axisVdifLB[c,Client,User] -53
 zV[c,lb,Client,User] DistAxisLB1[r,Client,User] 8 triangle1[Client,CertificateTemplate,User,c] 1
 zV[c,lb,Client,User] triangle1[Client,Course,User,c] 1 triangle1[Client,Role,User,c] 1
 zV[c,lb,Client,User] triangle2[Client,User,CourseSession,c] 1 triangle2[Client,User,Clinic,c] 1
 zV[c,lb,Client,User] triangle2[Client,User,Location,c] 1 triangle2[Client,User,Therapist,c] 1
 zV[c,lb,Client,User] triangle2[Client,User,Instructor,c] 1
 zV[r,rt,Client,CourseSession] choose1[Client,CourseSession] 1 axisVdifRT[r,Client,CourseSession] -51
 zV[r,rt,Client,CourseSession] DistAxisLB1[c,Client,CourseSession] 12.5 triangle1[Client,CourseSession,Clinic,r] 1
 zV[r,rt,Client,CourseSession] triangle1[Client,CourseSession,Location,r] 1 triangle1[Client,CourseSession,Therapist,r] 1
 zV[r,rt,Client,CourseSession] triangle1[Client,CourseSession,Instructor,r] 1 triangle2[Client,CertificateTemplate,CourseSession,r] 1
 zV[r,rt,Client,CourseSession] triangle2[Client,Course,CourseSession,r] 1 triangle2[Client,Role,CourseSession,r] 1
 zV[r,rt,Client,CourseSession] triangle2[Client,User,CourseSession,r] 1
 zV[c,rt,Client,CourseSession] choose1[Client,CourseSession] 1 axisVdifRT[c,Client,CourseSession] -56
 zV[c,rt,Client,CourseSession] DistAxisLB1[r,Client,CourseSession] 9 triangle1[Client,CourseSession,Clinic,c] 1
 zV[c,rt,Client,CourseSession] triangle1[Client,CourseSession,Location,c] 1 triangle1[Client,CourseSession,Therapist,c] 1
 zV[c,rt,Client,CourseSession] triangle1[Client,CourseSession,Instructor,c] 1 triangle2[Client,CertificateTemplate,CourseSession,c] 1
 zV[c,rt,Client,CourseSession] triangle2[Client,Course,CourseSession,c] 1 triangle2[Client,Role,CourseSession,c] 1
 zV[c,rt,Client,CourseSession] triangle2[Client,User,CourseSession,c] 1
 zV[r,lb,Client,CourseSession] choose1[Client,CourseSession] 1 axisVdifLB[r,Client,CourseSession] -51
 zV[r,lb,Client,CourseSession] DistAxisLB1[c,Client,CourseSession] 12.5 triangle1[Client,CertificateTemplate,CourseSession,r] 1
 zV[r,lb,Client,CourseSession] triangle1[Client,Course,CourseSession,r] 1 triangle1[Client,Role,CourseSession,r] 1
 zV[r,lb,Client,CourseSession] triangle1[Client,User,CourseSession,r] 1 triangle2[Client,CourseSession,Clinic,r] 1
 zV[r,lb,Client,CourseSession] triangle2[Client,CourseSession,Location,r] 1 triangle2[Client,CourseSession,Therapist,r] 1
 zV[r,lb,Client,CourseSession] triangle2[Client,CourseSession,Instructor,r] 1
 zV[c,lb,Client,CourseSession] choose1[Client,CourseSession] 1 axisVdifLB[c,Client,CourseSession] -53
 zV[c,lb,Client,CourseSession] DistAxisLB1[r,Client,CourseSession] 9 triangle1[Client,CertificateTemplate,CourseSession,c] 1
 zV[c,lb,Client,CourseSession] triangle1[Client,Course,CourseSession,c] 1 triangle1[Client,Role,CourseSession,c] 1
 zV[c,lb,Client,CourseSession] triangle1[Client,User,CourseSession,c] 1 triangle2[Client,CourseSession,Clinic,c] 1
 zV[c,lb,Client,CourseSession] triangle2[Client,CourseSession,Location,c] 1 triangle2[Client,CourseSession,Therapist,c] 1
 zV[c,lb,Client,CourseSession] triangle2[Client,CourseSession,Instructor,c] 1
 zV[r,rt,Client,Clinic] choose1[Client,Clinic] 1 axisVdifRT[r,Client,Clinic] -50
 zV[r,rt,Client,Clinic] triangle1[Client,Clinic,Location,r] 1 triangle1[Client,Clinic,Therapist,r] 1
 zV[r,rt,Client,Clinic] triangle1[Client,Clinic,Instructor,r] 1 triangle2[Client,CertificateTemplate,Clinic,r] 1
 zV[r,rt,Client,Clinic] triangle2[Client,Course,Clinic,r] 1 triangle2[Client,Role,Clinic,r] 1
 zV[r,rt,Client,Clinic] triangle2[Client,User,Clinic,r] 1 triangle2[Client,CourseSession,Clinic,r] 1
 zV[c,rt,Client,Clinic] choose1[Client,Clinic] 1 axisVdifRT[c,Client,Clinic] -53
 zV[c,rt,Client,Clinic] triangle1[Client,Clinic,Location,c] 1 triangle1[Client,Clinic,Therapist,c] 1
 zV[c,rt,Client,Clinic] triangle1[Client,Clinic,Instructor,c] 1 triangle2[Client,CertificateTemplate,Clinic,c] 1
 zV[c,rt,Client,Clinic] triangle2[Client,Course,Clinic,c] 1 triangle2[Client,Role,Clinic,c] 1
 zV[c,rt,Client,Clinic] triangle2[Client,User,Clinic,c] 1 triangle2[Client,CourseSession,Clinic,c] 1
 zV[r,lb,Client,Clinic] choose1[Client,Clinic] 1 axisVdifLB[r,Client,Clinic] -51
 zV[r,lb,Client,Clinic] triangle1[Client,CertificateTemplate,Clinic,r] 1 triangle1[Client,Course,Clinic,r] 1
 zV[r,lb,Client,Clinic] triangle1[Client,Role,Clinic,r] 1 triangle1[Client,User,Clinic,r] 1
 zV[r,lb,Client,Clinic] triangle1[Client,CourseSession,Clinic,r] 1 triangle2[Client,Clinic,Location,r] 1
 zV[r,lb,Client,Clinic] triangle2[Client,Clinic,Therapist,r] 1 triangle2[Client,Clinic,Instructor,r] 1
 zV[c,lb,Client,Clinic] choose1[Client,Clinic] 1 axisVdifLB[c,Client,Clinic] -53
 zV[c,lb,Client,Clinic] triangle1[Client,CertificateTemplate,Clinic,c] 1 triangle1[Client,Course,Clinic,c] 1
 zV[c,lb,Client,Clinic] triangle1[Client,Role,Clinic,c] 1 triangle1[Client,User,Clinic,c] 1
 zV[c,lb,Client,Clinic] triangle1[Client,CourseSession,Clinic,c] 1 triangle2[Client,Clinic,Location,c] 1
 zV[c,lb,Client,Clinic] triangle2[Client,Clinic,Therapist,c] 1 triangle2[Client,Clinic,Instructor,c] 1
 zV[r,rt,Client,Location] choose1[Client,Location] 1 axisVdifRT[r,Client,Location] -51
 zV[r,rt,Client,Location] DistAxisLB1[c,Client,Location] 11 triangle1[Client,Location,Therapist,r] 1
 zV[r,rt,Client,Location] triangle1[Client,Location,Instructor,r] 1 triangle2[Client,CertificateTemplate,Location,r] 1
 zV[r,rt,Client,Location] triangle2[Client,Course,Location,r] 1 triangle2[Client,Role,Location,r] 1
 zV[r,rt,Client,Location] triangle2[Client,User,Location,r] 1 triangle2[Client,CourseSession,Location,r] 1
 zV[r,rt,Client,Location] triangle2[Client,Clinic,Location,r] 1
 zV[c,rt,Client,Location] choose1[Client,Location] 1 axisVdifRT[c,Client,Location] -53
 zV[c,rt,Client,Location] DistAxisLB1[r,Client,Location] 9 triangle1[Client,Location,Therapist,c] 1
 zV[c,rt,Client,Location] triangle1[Client,Location,Instructor,c] 1 triangle2[Client,CertificateTemplate,Location,c] 1
 zV[c,rt,Client,Location] triangle2[Client,Course,Location,c] 1 triangle2[Client,Role,Location,c] 1
 zV[c,rt,Client,Location] triangle2[Client,User,Location,c] 1 triangle2[Client,CourseSession,Location,c] 1
 zV[c,rt,Client,Location] triangle2[Client,Clinic,Location,c] 1
 zV[r,lb,Client,Location] choose1[Client,Location] 1 axisVdifLB[r,Client,Location] -51
 zV[r,lb,Client,Location] DistAxisLB1[c,Client,Location] 11 triangle1[Client,CertificateTemplate,Location,r] 1
 zV[r,lb,Client,Location] triangle1[Client,Course,Location,r] 1 triangle1[Client,Role,Location,r] 1
 zV[r,lb,Client,Location] triangle1[Client,User,Location,r] 1 triangle1[Client,CourseSession,Location,r] 1
 zV[r,lb,Client,Location] triangle1[Client,Clinic,Location,r] 1 triangle2[Client,Location,Therapist,r] 1
 zV[r,lb,Client,Location] triangle2[Client,Location,Instructor,r] 1
 zV[c,lb,Client,Location] choose1[Client,Location] 1 axisVdifLB[c,Client,Location] -53
 zV[c,lb,Client,Location] DistAxisLB1[r,Client,Location] 9 triangle1[Client,CertificateTemplate,Location,c] 1
 zV[c,lb,Client,Location] triangle1[Client,Course,Location,c] 1 triangle1[Client,Role,Location,c] 1
 zV[c,lb,Client,Location] triangle1[Client,User,Location,c] 1 triangle1[Client,CourseSession,Location,c] 1
 zV[c,lb,Client,Location] triangle1[Client,Clinic,Location,c] 1 triangle2[Client,Location,Therapist,c] 1
 zV[c,lb,Client,Location] triangle2[Client,Location,Instructor,c] 1
 zV[r,rt,Client,Therapist] choose1[Client,Therapist] 1 axisVdifRT[r,Client,Therapist] -52
 zV[r,rt,Client,Therapist] triangle1[Client,Therapist,Instructor,r] 1 triangle2[Client,CertificateTemplate,Therapist,r] 1
 zV[r,rt,Client,Therapist] triangle2[Client,Course,Therapist,r] 1 triangle2[Client,Role,Therapist,r] 1
 zV[r,rt,Client,Therapist] triangle2[Client,User,Therapist,r] 1 triangle2[Client,CourseSession,Therapist,r] 1
 zV[r,rt,Client,Therapist] triangle2[Client,Clinic,Therapist,r] 1 triangle2[Client,Location,Therapist,r] 1
 zV[c,rt,Client,Therapist] choose1[Client,Therapist] 1 axisVdifRT[c,Client,Therapist] -53
 zV[c,rt,Client,Therapist] triangle1[Client,Therapist,Instructor,c] 1 triangle2[Client,CertificateTemplate,Therapist,c] 1
 zV[c,rt,Client,Therapist] triangle2[Client,Course,Therapist,c] 1 triangle2[Client,Role,Therapist,c] 1
 zV[c,rt,Client,Therapist] triangle2[Client,User,Therapist,c] 1 triangle2[Client,CourseSession,Therapist,c] 1
 zV[c,rt,Client,Therapist] triangle2[Client,Clinic,Therapist,c] 1 triangle2[Client,Location,Therapist,c] 1
 zV[r,lb,Client,Therapist] choose1[Client,Therapist] 1 axisVdifLB[r,Client,Therapist] -51
 zV[r,lb,Client,Therapist] triangle1[Client,CertificateTemplate,Therapist,r] 1 triangle1[Client,Course,Therapist,r] 1
 zV[r,lb,Client,Therapist] triangle1[Client,Role,Therapist,r] 1 triangle1[Client,User,Therapist,r] 1
 zV[r,lb,Client,Therapist] triangle1[Client,CourseSession,Therapist,r] 1 triangle1[Client,Clinic,Therapist,r] 1
 zV[r,lb,Client,Therapist] triangle1[Client,Location,Therapist,r] 1 triangle2[Client,Therapist,Instructor,r] 1
 zV[c,lb,Client,Therapist] choose1[Client,Therapist] 1 axisVdifLB[c,Client,Therapist] -53
 zV[c,lb,Client,Therapist] triangle1[Client,CertificateTemplate,Therapist,c] 1 triangle1[Client,Course,Therapist,c] 1
 zV[c,lb,Client,Therapist] triangle1[Client,Role,Therapist,c] 1 triangle1[Client,User,Therapist,c] 1
 zV[c,lb,Client,Therapist] triangle1[Client,CourseSession,Therapist,c] 1 triangle1[Client,Clinic,Therapist,c] 1
 zV[c,lb,Client,Therapist] triangle1[Client,Location,Therapist,c] 1 triangle2[Client,Therapist,Instructor,c] 1
 zV[r,rt,Client,Instructor] choose1[Client,Instructor] 1 axisVdifRT[r,Client,Instructor] -52
 zV[r,rt,Client,Instructor] DistAxisLB1[c,Client,Instructor] 11.5 triangle2[Client,CertificateTemplate,Instructor,r] 1
 zV[r,rt,Client,Instructor] triangle2[Client,Course,Instructor,r] 1 triangle2[Client,Role,Instructor,r] 1
 zV[r,rt,Client,Instructor] triangle2[Client,User,Instructor,r] 1 triangle2[Client,CourseSession,Instructor,r] 1
 zV[r,rt,Client,Instructor] triangle2[Client,Clinic,Instructor,r] 1 triangle2[Client,Location,Instructor,r] 1
 zV[r,rt,Client,Instructor] triangle2[Client,Therapist,Instructor,r] 1
 zV[c,rt,Client,Instructor] choose1[Client,Instructor] 1 axisVdifRT[c,Client,Instructor] -54
 zV[c,rt,Client,Instructor] DistAxisLB1[r,Client,Instructor] 9.5 triangle2[Client,CertificateTemplate,Instructor,c] 1
 zV[c,rt,Client,Instructor] triangle2[Client,Course,Instructor,c] 1 triangle2[Client,Role,Instructor,c] 1
 zV[c,rt,Client,Instructor] triangle2[Client,User,Instructor,c] 1 triangle2[Client,CourseSession,Instructor,c] 1
 zV[c,rt,Client,Instructor] triangle2[Client,Clinic,Instructor,c] 1 triangle2[Client,Location,Instructor,c] 1
 zV[c,rt,Client,Instructor] triangle2[Client,Therapist,Instructor,c] 1
 zV[r,lb,Client,Instructor] choose1[Client,Instructor] 1 axisVdifLB[r,Client,Instructor] -51
 zV[r,lb,Client,Instructor] DistAxisLB1[c,Client,Instructor] 11.5 triangle1[Client,CertificateTemplate,Instructor,r] 1
 zV[r,lb,Client,Instructor] triangle1[Client,Course,Instructor,r] 1 triangle1[Client,Role,Instructor,r] 1
 zV[r,lb,Client,Instructor] triangle1[Client,User,Instructor,r] 1 triangle1[Client,CourseSession,Instructor,r] 1
 zV[r,lb,Client,Instructor] triangle1[Client,Clinic,Instructor,r] 1 triangle1[Client,Location,Instructor,r] 1
 zV[r,lb,Client,Instructor] triangle1[Client,Therapist,Instructor,r] 1
 zV[c,lb,Client,Instructor] choose1[Client,Instructor] 1 axisVdifLB[c,Client,Instructor] -53
 zV[c,lb,Client,Instructor] DistAxisLB1[r,Client,Instructor] 9.5 triangle1[Client,CertificateTemplate,Instructor,c] 1
 zV[c,lb,Client,Instructor] triangle1[Client,Course,Instructor,c] 1 triangle1[Client,Role,Instructor,c] 1
 zV[c,lb,Client,Instructor] triangle1[Client,User,Instructor,c] 1 triangle1[Client,CourseSession,Instructor,c] 1
 zV[c,lb,Client,Instructor] triangle1[Client,Clinic,Instructor,c] 1 triangle1[Client,Location,Instructor,c] 1
 zV[c,lb,Client,Instructor] triangle1[Client,Therapist,Instructor,c] 1
 zV[r,rt,CertificateTemplate,Course] choose1[CertificateTemplate,Course] 1 axisVdifRT[r,CertificateTemplate,Course] -48
 zV[r,rt,CertificateTemplate,Course] triangle1[Client,CertificateTemplate,Course,r] 1 triangle1[CertificateTemplate,Course,Role,r] 1
 zV[r,rt,CertificateTemplate,Course] triangle1[CertificateTemplate,Course,User,r] 1 triangle1[CertificateTemplate,Course,CourseSession,r] 1
 zV[r,rt,CertificateTemplate,Course] triangle1[CertificateTemplate,Course,Clinic,r] 1 triangle1[CertificateTemplate,Course,Location,r] 1
 zV[r,rt,CertificateTemplate,Course] triangle1[CertificateTemplate,Course,Therapist,r] 1 triangle1[CertificateTemplate,Course,Instructor,r] 1
 zV[c,rt,CertificateTemplate,Course] choose1[CertificateTemplate,Course] 1 axisVdifRT[c,CertificateTemplate,Course] -52
 zV[c,rt,CertificateTemplate,Course] triangle1[Client,CertificateTemplate,Course,c] 1 triangle1[CertificateTemplate,Course,Role,c] 1
 zV[c,rt,CertificateTemplate,Course] triangle1[CertificateTemplate,Course,User,c] 1 triangle1[CertificateTemplate,Course,CourseSession,c] 1
 zV[c,rt,CertificateTemplate,Course] triangle1[CertificateTemplate,Course,Clinic,c] 1 triangle1[CertificateTemplate,Course,Location,c] 1
 zV[c,rt,CertificateTemplate,Course] triangle1[CertificateTemplate,Course,Therapist,c] 1 triangle1[CertificateTemplate,Course,Instructor,c] 1
 zV[r,lb,CertificateTemplate,Course] choose1[CertificateTemplate,Course] 1 axisVdifLB[r,CertificateTemplate,Course] -49
 zV[r,lb,CertificateTemplate,Course] triangle2[Client,CertificateTemplate,Course,r] 1 triangle2[CertificateTemplate,Course,Role,r] 1
 zV[r,lb,CertificateTemplate,Course] triangle2[CertificateTemplate,Course,User,r] 1 triangle2[CertificateTemplate,Course,CourseSession,r] 1
 zV[r,lb,CertificateTemplate,Course] triangle2[CertificateTemplate,Course,Clinic,r] 1 triangle2[CertificateTemplate,Course,Location,r] 1
 zV[r,lb,CertificateTemplate,Course] triangle2[CertificateTemplate,Course,Therapist,r] 1 triangle2[CertificateTemplate,Course,Instructor,r] 1
 zV[c,lb,CertificateTemplate,Course] choose1[CertificateTemplate,Course] 1 axisVdifLB[c,CertificateTemplate,Course] -54
 zV[c,lb,CertificateTemplate,Course] triangle2[Client,CertificateTemplate,Course,c] 1 triangle2[CertificateTemplate,Course,Role,c] 1
 zV[c,lb,CertificateTemplate,Course] triangle2[CertificateTemplate,Course,User,c] 1 triangle2[CertificateTemplate,Course,CourseSession,c] 1
 zV[c,lb,CertificateTemplate,Course] triangle2[CertificateTemplate,Course,Clinic,c] 1 triangle2[CertificateTemplate,Course,Location,c] 1
 zV[c,lb,CertificateTemplate,Course] triangle2[CertificateTemplate,Course,Therapist,c] 1 triangle2[CertificateTemplate,Course,Instructor,c] 1
 zV[r,rt,CertificateTemplate,Role] choose1[CertificateTemplate,Role] 1 axisVdifRT[r,CertificateTemplate,Role] -48
 zV[r,rt,CertificateTemplate,Role] triangle1[Client,CertificateTemplate,Role,r] 1 triangle1[CertificateTemplate,Role,User,r] 1
 zV[r,rt,CertificateTemplate,Role] triangle1[CertificateTemplate,Role,CourseSession,r] 1 triangle1[CertificateTemplate,Role,Clinic,r] 1
 zV[r,rt,CertificateTemplate,Role] triangle1[CertificateTemplate,Role,Location,r] 1 triangle1[CertificateTemplate,Role,Therapist,r] 1
 zV[r,rt,CertificateTemplate,Role] triangle1[CertificateTemplate,Role,Instructor,r] 1 triangle2[CertificateTemplate,Course,Role,r] 1
 zV[c,rt,CertificateTemplate,Role] choose1[CertificateTemplate,Role] 1 axisVdifRT[c,CertificateTemplate,Role] -52
 zV[c,rt,CertificateTemplate,Role] triangle1[Client,CertificateTemplate,Role,c] 1 triangle1[CertificateTemplate,Role,User,c] 1
 zV[c,rt,CertificateTemplate,Role] triangle1[CertificateTemplate,Role,CourseSession,c] 1 triangle1[CertificateTemplate,Role,Clinic,c] 1
 zV[c,rt,CertificateTemplate,Role] triangle1[CertificateTemplate,Role,Location,c] 1 triangle1[CertificateTemplate,Role,Therapist,c] 1
 zV[c,rt,CertificateTemplate,Role] triangle1[CertificateTemplate,Role,Instructor,c] 1 triangle2[CertificateTemplate,Course,Role,c] 1
 zV[r,lb,CertificateTemplate,Role] choose1[CertificateTemplate,Role] 1 axisVdifLB[r,CertificateTemplate,Role] -49
 zV[r,lb,CertificateTemplate,Role] triangle1[CertificateTemplate,Course,Role,r] 1 triangle2[Client,CertificateTemplate,Role,r] 1
 zV[r,lb,CertificateTemplate,Role] triangle2[CertificateTemplate,Role,User,r] 1 triangle2[CertificateTemplate,Role,CourseSession,r] 1
 zV[r,lb,CertificateTemplate,Role] triangle2[CertificateTemplate,Role,Clinic,r] 1 triangle2[CertificateTemplate,Role,Location,r] 1
 zV[r,lb,CertificateTemplate,Role] triangle2[CertificateTemplate,Role,Therapist,r] 1 triangle2[CertificateTemplate,Role,Instructor,r] 1
 zV[c,lb,CertificateTemplate,Role] choose1[CertificateTemplate,Role] 1 axisVdifLB[c,CertificateTemplate,Role] -54
 zV[c,lb,CertificateTemplate,Role] triangle1[CertificateTemplate,Course,Role,c] 1 triangle2[Client,CertificateTemplate,Role,c] 1
 zV[c,lb,CertificateTemplate,Role] triangle2[CertificateTemplate,Role,User,c] 1 triangle2[CertificateTemplate,Role,CourseSession,c] 1
 zV[c,lb,CertificateTemplate,Role] triangle2[CertificateTemplate,Role,Clinic,c] 1 triangle2[CertificateTemplate,Role,Location,c] 1
 zV[c,lb,CertificateTemplate,Role] triangle2[CertificateTemplate,Role,Therapist,c] 1 triangle2[CertificateTemplate,Role,Instructor,c] 1
 zV[r,rt,CertificateTemplate,User] choose1[CertificateTemplate,User] 1 axisVdifRT[r,CertificateTemplate,User] -49
 zV[r,rt,CertificateTemplate,User] triangle1[Client,CertificateTemplate,User,r] 1 triangle1[CertificateTemplate,User,CourseSession,r] 1
 zV[r,rt,CertificateTemplate,User] triangle1[CertificateTemplate,User,Clinic,r] 1 triangle1[CertificateTemplate,User,Location,r] 1
 zV[r,rt,CertificateTemplate,User] triangle1[CertificateTemplate,User,Therapist,r] 1 triangle1[CertificateTemplate,User,Instructor,r] 1
 zV[r,rt,CertificateTemplate,User] triangle2[CertificateTemplate,Course,User,r] 1 triangle2[CertificateTemplate,Role,User,r] 1
 zV[c,rt,CertificateTemplate,User] choose1[CertificateTemplate,User] 1 axisVdifRT[c,CertificateTemplate,User] -53
 zV[c,rt,CertificateTemplate,User] triangle1[Client,CertificateTemplate,User,c] 1 triangle1[CertificateTemplate,User,CourseSession,c] 1
 zV[c,rt,CertificateTemplate,User] triangle1[CertificateTemplate,User,Clinic,c] 1 triangle1[CertificateTemplate,User,Location,c] 1
 zV[c,rt,CertificateTemplate,User] triangle1[CertificateTemplate,User,Therapist,c] 1 triangle1[CertificateTemplate,User,Instructor,c] 1
 zV[c,rt,CertificateTemplate,User] triangle2[CertificateTemplate,Course,User,c] 1 triangle2[CertificateTemplate,Role,User,c] 1
 zV[r,lb,CertificateTemplate,User] choose1[CertificateTemplate,User] 1 axisVdifLB[r,CertificateTemplate,User] -49
 zV[r,lb,CertificateTemplate,User] triangle1[CertificateTemplate,Course,User,r] 1 triangle1[CertificateTemplate,Role,User,r] 1
 zV[r,lb,CertificateTemplate,User] triangle2[Client,CertificateTemplate,User,r] 1 triangle2[CertificateTemplate,User,CourseSession,r] 1
 zV[r,lb,CertificateTemplate,User] triangle2[CertificateTemplate,User,Clinic,r] 1 triangle2[CertificateTemplate,User,Location,r] 1
 zV[r,lb,CertificateTemplate,User] triangle2[CertificateTemplate,User,Therapist,r] 1 triangle2[CertificateTemplate,User,Instructor,r] 1
 zV[c,lb,CertificateTemplate,User] choose1[CertificateTemplate,User] 1 axisVdifLB[c,CertificateTemplate,User] -54
 zV[c,lb,CertificateTemplate,User] triangle1[CertificateTemplate,Course,User,c] 1 triangle1[CertificateTemplate,Role,User,c] 1
 zV[c,lb,CertificateTemplate,User] triangle2[Client,CertificateTemplate,User,c] 1 triangle2[CertificateTemplate,User,CourseSession,c] 1
 zV[c,lb,CertificateTemplate,User] triangle2[CertificateTemplate,User,Clinic,c] 1 triangle2[CertificateTemplate,User,Location,c] 1
 zV[c,lb,CertificateTemplate,User] triangle2[CertificateTemplate,User,Therapist,c] 1 triangle2[CertificateTemplate,User,Instructor,c] 1
 zV[r,rt,CertificateTemplate,CourseSession] choose1[CertificateTemplate,CourseSession] 1 axisVdifRT[r,CertificateTemplate,CourseSession] -51
 zV[r,rt,CertificateTemplate,CourseSession] DistAxisLB1[c,CertificateTemplate,CourseSession] 13 triangle1[Client,CertificateTemplate,CourseSession,r] 1
 zV[r,rt,CertificateTemplate,CourseSession] triangle1[CertificateTemplate,CourseSession,Clinic,r] 1 triangle1[CertificateTemplate,CourseSession,Location,r] 1
 zV[r,rt,CertificateTemplate,CourseSession] triangle1[CertificateTemplate,CourseSession,Therapist,r] 1 triangle1[CertificateTemplate,CourseSession,Instructor,r] 1
 zV[r,rt,CertificateTemplate,CourseSession] triangle2[CertificateTemplate,Course,CourseSession,r] 1 triangle2[CertificateTemplate,Role,CourseSession,r] 1
 zV[r,rt,CertificateTemplate,CourseSession] triangle2[CertificateTemplate,User,CourseSession,r] 1
 zV[c,rt,CertificateTemplate,CourseSession] choose1[CertificateTemplate,CourseSession] 1 axisVdifRT[c,CertificateTemplate,CourseSession] -56
 zV[c,rt,CertificateTemplate,CourseSession] DistAxisLB1[r,CertificateTemplate,CourseSession] 8 triangle1[Client,CertificateTemplate,CourseSession,c] 1
 zV[c,rt,CertificateTemplate,CourseSession] triangle1[CertificateTemplate,CourseSession,Clinic,c] 1 triangle1[CertificateTemplate,CourseSession,Location,c] 1
 zV[c,rt,CertificateTemplate,CourseSession] triangle1[CertificateTemplate,CourseSession,Therapist,c] 1 triangle1[CertificateTemplate,CourseSession,Instructor,c] 1
 zV[c,rt,CertificateTemplate,CourseSession] triangle2[CertificateTemplate,Course,CourseSession,c] 1 triangle2[CertificateTemplate,Role,CourseSession,c] 1
 zV[c,rt,CertificateTemplate,CourseSession] triangle2[CertificateTemplate,User,CourseSession,c] 1
 zV[r,lb,CertificateTemplate,CourseSession] choose1[CertificateTemplate,CourseSession] 1 axisVdifLB[r,CertificateTemplate,CourseSession] -49
 zV[r,lb,CertificateTemplate,CourseSession] DistAxisLB1[c,CertificateTemplate,CourseSession] 13 triangle1[CertificateTemplate,Course,CourseSession,r] 1
 zV[r,lb,CertificateTemplate,CourseSession] triangle1[CertificateTemplate,Role,CourseSession,r] 1 triangle1[CertificateTemplate,User,CourseSession,r] 1
 zV[r,lb,CertificateTemplate,CourseSession] triangle2[Client,CertificateTemplate,CourseSession,r] 1 triangle2[CertificateTemplate,CourseSession,Clinic,r] 1
 zV[r,lb,CertificateTemplate,CourseSession] triangle2[CertificateTemplate,CourseSession,Location,r] 1 triangle2[CertificateTemplate,CourseSession,Therapist,r] 1
 zV[r,lb,CertificateTemplate,CourseSession] triangle2[CertificateTemplate,CourseSession,Instructor,r] 1
 zV[c,lb,CertificateTemplate,CourseSession] choose1[CertificateTemplate,CourseSession] 1 axisVdifLB[c,CertificateTemplate,CourseSession] -54
 zV[c,lb,CertificateTemplate,CourseSession] DistAxisLB1[r,CertificateTemplate,CourseSession] 8 triangle1[CertificateTemplate,Course,CourseSession,c] 1
 zV[c,lb,CertificateTemplate,CourseSession] triangle1[CertificateTemplate,Role,CourseSession,c] 1 triangle1[CertificateTemplate,User,CourseSession,c] 1
 zV[c,lb,CertificateTemplate,CourseSession] triangle2[Client,CertificateTemplate,CourseSession,c] 1 triangle2[CertificateTemplate,CourseSession,Clinic,c] 1
 zV[c,lb,CertificateTemplate,CourseSession] triangle2[CertificateTemplate,CourseSession,Location,c] 1 triangle2[CertificateTemplate,CourseSession,Therapist,c] 1
 zV[c,lb,CertificateTemplate,CourseSession] triangle2[CertificateTemplate,CourseSession,Instructor,c] 1
 zV[r,rt,CertificateTemplate,Clinic] choose1[CertificateTemplate,Clinic] 1 axisVdifRT[r,CertificateTemplate,Clinic] -50
 zV[r,rt,CertificateTemplate,Clinic] triangle1[Client,CertificateTemplate,Clinic,r] 1 triangle1[CertificateTemplate,Clinic,Location,r] 1
 zV[r,rt,CertificateTemplate,Clinic] triangle1[CertificateTemplate,Clinic,Therapist,r] 1 triangle1[CertificateTemplate,Clinic,Instructor,r] 1
 zV[r,rt,CertificateTemplate,Clinic] triangle2[CertificateTemplate,Course,Clinic,r] 1 triangle2[CertificateTemplate,Role,Clinic,r] 1
 zV[r,rt,CertificateTemplate,Clinic] triangle2[CertificateTemplate,User,Clinic,r] 1 triangle2[CertificateTemplate,CourseSession,Clinic,r] 1
 zV[c,rt,CertificateTemplate,Clinic] choose1[CertificateTemplate,Clinic] 1 axisVdifRT[c,CertificateTemplate,Clinic] -53
 zV[c,rt,CertificateTemplate,Clinic] triangle1[Client,CertificateTemplate,Clinic,c] 1 triangle1[CertificateTemplate,Clinic,Location,c] 1
 zV[c,rt,CertificateTemplate,Clinic] triangle1[CertificateTemplate,Clinic,Therapist,c] 1 triangle1[CertificateTemplate,Clinic,Instructor,c] 1
 zV[c,rt,CertificateTemplate,Clinic] triangle2[CertificateTemplate,Course,Clinic,c] 1 triangle2[CertificateTemplate,Role,Clinic,c] 1
 zV[c,rt,CertificateTemplate,Clinic] triangle2[CertificateTemplate,User,Clinic,c] 1 triangle2[CertificateTemplate,CourseSession,Clinic,c] 1
 zV[r,lb,CertificateTemplate,Clinic] choose1[CertificateTemplate,Clinic] 1 axisVdifLB[r,CertificateTemplate,Clinic] -49
 zV[r,lb,CertificateTemplate,Clinic] triangle1[CertificateTemplate,Course,Clinic,r] 1 triangle1[CertificateTemplate,Role,Clinic,r] 1
 zV[r,lb,CertificateTemplate,Clinic] triangle1[CertificateTemplate,User,Clinic,r] 1 triangle1[CertificateTemplate,CourseSession,Clinic,r] 1
 zV[r,lb,CertificateTemplate,Clinic] triangle2[Client,CertificateTemplate,Clinic,r] 1 triangle2[CertificateTemplate,Clinic,Location,r] 1
 zV[r,lb,CertificateTemplate,Clinic] triangle2[CertificateTemplate,Clinic,Therapist,r] 1 triangle2[CertificateTemplate,Clinic,Instructor,r] 1
 zV[c,lb,CertificateTemplate,Clinic] choose1[CertificateTemplate,Clinic] 1 axisVdifLB[c,CertificateTemplate,Clinic] -54
 zV[c,lb,CertificateTemplate,Clinic] triangle1[CertificateTemplate,Course,Clinic,c] 1 triangle1[CertificateTemplate,Role,Clinic,c] 1
 zV[c,lb,CertificateTemplate,Clinic] triangle1[CertificateTemplate,User,Clinic,c] 1 triangle1[CertificateTemplate,CourseSession,Clinic,c] 1
 zV[c,lb,CertificateTemplate,Clinic] triangle2[Client,CertificateTemplate,Clinic,c] 1 triangle2[CertificateTemplate,Clinic,Location,c] 1
 zV[c,lb,CertificateTemplate,Clinic] triangle2[CertificateTemplate,Clinic,Therapist,c] 1 triangle2[CertificateTemplate,Clinic,Instructor,c] 1
 zV[r,rt,CertificateTemplate,Location] choose1[CertificateTemplate,Location] 1 axisVdifRT[r,CertificateTemplate,Location] -51
 zV[r,rt,CertificateTemplate,Location] triangle1[Client,CertificateTemplate,Location,r] 1 triangle1[CertificateTemplate,Location,Therapist,r] 1
 zV[r,rt,CertificateTemplate,Location] triangle1[CertificateTemplate,Location,Instructor,r] 1 triangle2[CertificateTemplate,Course,Location,r] 1
 zV[r,rt,CertificateTemplate,Location] triangle2[CertificateTemplate,Role,Location,r] 1 triangle2[CertificateTemplate,User,Location,r] 1
 zV[r,rt,CertificateTemplate,Location] triangle2[CertificateTemplate,CourseSession,Location,r] 1 triangle2[CertificateTemplate,Clinic,Location,r] 1
 zV[c,rt,CertificateTemplate,Location] choose1[CertificateTemplate,Location] 1 axisVdifRT[c,CertificateTemplate,Location] -53
 zV[c,rt,CertificateTemplate,Location] triangle1[Client,CertificateTemplate,Location,c] 1 triangle1[CertificateTemplate,Location,Therapist,c] 1
 zV[c,rt,CertificateTemplate,Location] triangle1[CertificateTemplate,Location,Instructor,c] 1 triangle2[CertificateTemplate,Course,Location,c] 1
 zV[c,rt,CertificateTemplate,Location] triangle2[CertificateTemplate,Role,Location,c] 1 triangle2[CertificateTemplate,User,Location,c] 1
 zV[c,rt,CertificateTemplate,Location] triangle2[CertificateTemplate,CourseSession,Location,c] 1 triangle2[CertificateTemplate,Clinic,Location,c] 1
 zV[r,lb,CertificateTemplate,Location] choose1[CertificateTemplate,Location] 1 axisVdifLB[r,CertificateTemplate,Location] -49
 zV[r,lb,CertificateTemplate,Location] triangle1[CertificateTemplate,Course,Location,r] 1 triangle1[CertificateTemplate,Role,Location,r] 1
 zV[r,lb,CertificateTemplate,Location] triangle1[CertificateTemplate,User,Location,r] 1 triangle1[CertificateTemplate,CourseSession,Location,r] 1
 zV[r,lb,CertificateTemplate,Location] triangle1[CertificateTemplate,Clinic,Location,r] 1 triangle2[Client,CertificateTemplate,Location,r] 1
 zV[r,lb,CertificateTemplate,Location] triangle2[CertificateTemplate,Location,Therapist,r] 1 triangle2[CertificateTemplate,Location,Instructor,r] 1
 zV[c,lb,CertificateTemplate,Location] choose1[CertificateTemplate,Location] 1 axisVdifLB[c,CertificateTemplate,Location] -54
 zV[c,lb,CertificateTemplate,Location] triangle1[CertificateTemplate,Course,Location,c] 1 triangle1[CertificateTemplate,Role,Location,c] 1
 zV[c,lb,CertificateTemplate,Location] triangle1[CertificateTemplate,User,Location,c] 1 triangle1[CertificateTemplate,CourseSession,Location,c] 1
 zV[c,lb,CertificateTemplate,Location] triangle1[CertificateTemplate,Clinic,Location,c] 1 triangle2[Client,CertificateTemplate,Location,c] 1
 zV[c,lb,CertificateTemplate,Location] triangle2[CertificateTemplate,Location,Therapist,c] 1 triangle2[CertificateTemplate,Location,Instructor,c] 1
 zV[r,rt,CertificateTemplate,Therapist] choose1[CertificateTemplate,Therapist] 1 axisVdifRT[r,CertificateTemplate,Therapist] -52
 zV[r,rt,CertificateTemplate,Therapist] triangle1[Client,CertificateTemplate,Therapist,r] 1 triangle1[CertificateTemplate,Therapist,Instructor,r] 1
 zV[r,rt,CertificateTemplate,Therapist] triangle2[CertificateTemplate,Course,Therapist,r] 1 triangle2[CertificateTemplate,Role,Therapist,r] 1
 zV[r,rt,CertificateTemplate,Therapist] triangle2[CertificateTemplate,User,Therapist,r] 1 triangle2[CertificateTemplate,CourseSession,Therapist,r] 1
 zV[r,rt,CertificateTemplate,Therapist] triangle2[CertificateTemplate,Clinic,Therapist,r] 1 triangle2[CertificateTemplate,Location,Therapist,r] 1
 zV[c,rt,CertificateTemplate,Therapist] choose1[CertificateTemplate,Therapist] 1 axisVdifRT[c,CertificateTemplate,Therapist] -53
 zV[c,rt,CertificateTemplate,Therapist] triangle1[Client,CertificateTemplate,Therapist,c] 1 triangle1[CertificateTemplate,Therapist,Instructor,c] 1
 zV[c,rt,CertificateTemplate,Therapist] triangle2[CertificateTemplate,Course,Therapist,c] 1 triangle2[CertificateTemplate,Role,Therapist,c] 1
 zV[c,rt,CertificateTemplate,Therapist] triangle2[CertificateTemplate,User,Therapist,c] 1 triangle2[CertificateTemplate,CourseSession,Therapist,c] 1
 zV[c,rt,CertificateTemplate,Therapist] triangle2[CertificateTemplate,Clinic,Therapist,c] 1 triangle2[CertificateTemplate,Location,Therapist,c] 1
 zV[r,lb,CertificateTemplate,Therapist] choose1[CertificateTemplate,Therapist] 1 axisVdifLB[r,CertificateTemplate,Therapist] -49
 zV[r,lb,CertificateTemplate,Therapist] triangle1[CertificateTemplate,Course,Therapist,r] 1 triangle1[CertificateTemplate,Role,Therapist,r] 1
 zV[r,lb,CertificateTemplate,Therapist] triangle1[CertificateTemplate,User,Therapist,r] 1 triangle1[CertificateTemplate,CourseSession,Therapist,r] 1
 zV[r,lb,CertificateTemplate,Therapist] triangle1[CertificateTemplate,Clinic,Therapist,r] 1 triangle1[CertificateTemplate,Location,Therapist,r] 1
 zV[r,lb,CertificateTemplate,Therapist] triangle2[Client,CertificateTemplate,Therapist,r] 1 triangle2[CertificateTemplate,Therapist,Instructor,r] 1
 zV[c,lb,CertificateTemplate,Therapist] choose1[CertificateTemplate,Therapist] 1 axisVdifLB[c,CertificateTemplate,Therapist] -54
 zV[c,lb,CertificateTemplate,Therapist] triangle1[CertificateTemplate,Course,Therapist,c] 1 triangle1[CertificateTemplate,Role,Therapist,c] 1
 zV[c,lb,CertificateTemplate,Therapist] triangle1[CertificateTemplate,User,Therapist,c] 1 triangle1[CertificateTemplate,CourseSession,Therapist,c] 1
 zV[c,lb,CertificateTemplate,Therapist] triangle1[CertificateTemplate,Clinic,Therapist,c] 1 triangle1[CertificateTemplate,Location,Therapist,c] 1
 zV[c,lb,CertificateTemplate,Therapist] triangle2[Client,CertificateTemplate,Therapist,c] 1 triangle2[CertificateTemplate,Therapist,Instructor,c] 1
 zV[r,rt,CertificateTemplate,Instructor] choose1[CertificateTemplate,Instructor] 1 axisVdifRT[r,CertificateTemplate,Instructor] -52
 zV[r,rt,CertificateTemplate,Instructor] triangle1[Client,CertificateTemplate,Instructor,r] 1 triangle2[CertificateTemplate,Course,Instructor,r] 1
 zV[r,rt,CertificateTemplate,Instructor] triangle2[CertificateTemplate,Role,Instructor,r] 1 triangle2[CertificateTemplate,User,Instructor,r] 1
 zV[r,rt,CertificateTemplate,Instructor] triangle2[CertificateTemplate,CourseSession,Instructor,r] 1 triangle2[CertificateTemplate,Clinic,Instructor,r] 1
 zV[r,rt,CertificateTemplate,Instructor] triangle2[CertificateTemplate,Location,Instructor,r] 1 triangle2[CertificateTemplate,Therapist,Instructor,r] 1
 zV[c,rt,CertificateTemplate,Instructor] choose1[CertificateTemplate,Instructor] 1 axisVdifRT[c,CertificateTemplate,Instructor] -54
 zV[c,rt,CertificateTemplate,Instructor] triangle1[Client,CertificateTemplate,Instructor,c] 1 triangle2[CertificateTemplate,Course,Instructor,c] 1
 zV[c,rt,CertificateTemplate,Instructor] triangle2[CertificateTemplate,Role,Instructor,c] 1 triangle2[CertificateTemplate,User,Instructor,c] 1
 zV[c,rt,CertificateTemplate,Instructor] triangle2[CertificateTemplate,CourseSession,Instructor,c] 1 triangle2[CertificateTemplate,Clinic,Instructor,c] 1
 zV[c,rt,CertificateTemplate,Instructor] triangle2[CertificateTemplate,Location,Instructor,c] 1 triangle2[CertificateTemplate,Therapist,Instructor,c] 1
 zV[r,lb,CertificateTemplate,Instructor] choose1[CertificateTemplate,Instructor] 1 axisVdifLB[r,CertificateTemplate,Instructor] -49
 zV[r,lb,CertificateTemplate,Instructor] triangle1[CertificateTemplate,Course,Instructor,r] 1 triangle1[CertificateTemplate,Role,Instructor,r] 1
 zV[r,lb,CertificateTemplate,Instructor] triangle1[CertificateTemplate,User,Instructor,r] 1 triangle1[CertificateTemplate,CourseSession,Instructor,r] 1
 zV[r,lb,CertificateTemplate,Instructor] triangle1[CertificateTemplate,Clinic,Instructor,r] 1 triangle1[CertificateTemplate,Location,Instructor,r] 1
 zV[r,lb,CertificateTemplate,Instructor] triangle1[CertificateTemplate,Therapist,Instructor,r] 1 triangle2[Client,CertificateTemplate,Instructor,r] 1
 zV[c,lb,CertificateTemplate,Instructor] choose1[CertificateTemplate,Instructor] 1 axisVdifLB[c,CertificateTemplate,Instructor] -54
 zV[c,lb,CertificateTemplate,Instructor] triangle1[CertificateTemplate,Course,Instructor,c] 1 triangle1[CertificateTemplate,Role,Instructor,c] 1
 zV[c,lb,CertificateTemplate,Instructor] triangle1[CertificateTemplate,User,Instructor,c] 1 triangle1[CertificateTemplate,CourseSession,Instructor,c] 1
 zV[c,lb,CertificateTemplate,Instructor] triangle1[CertificateTemplate,Clinic,Instructor,c] 1 triangle1[CertificateTemplate,Location,Instructor,c] 1
 zV[c,lb,CertificateTemplate,Instructor] triangle1[CertificateTemplate,Therapist,Instructor,c] 1 triangle2[Client,CertificateTemplate,Instructor,c] 1
 zV[r,rt,Course,Role] choose1[Course,Role] 1 axisVdifRT[r,Course,Role] -48
 zV[r,rt,Course,Role] triangle1[Client,Course,Role,r] 1 triangle1[CertificateTemplate,Course,Role,r] 1
 zV[r,rt,Course,Role] triangle1[Course,Role,User,r] 1 triangle1[Course,Role,CourseSession,r] 1
 zV[r,rt,Course,Role] triangle1[Course,Role,Clinic,r] 1 triangle1[Course,Role,Location,r] 1
 zV[r,rt,Course,Role] triangle1[Course,Role,Therapist,r] 1 triangle1[Course,Role,Instructor,r] 1
 zV[c,rt,Course,Role] choose1[Course,Role] 1 axisVdifRT[c,Course,Role] -52
 zV[c,rt,Course,Role] triangle1[Client,Course,Role,c] 1 triangle1[CertificateTemplate,Course,Role,c] 1
 zV[c,rt,Course,Role] triangle1[Course,Role,User,c] 1 triangle1[Course,Role,CourseSession,c] 1
 zV[c,rt,Course,Role] triangle1[Course,Role,Clinic,c] 1 triangle1[Course,Role,Location,c] 1
 zV[c,rt,Course,Role] triangle1[Course,Role,Therapist,c] 1 triangle1[Course,Role,Instructor,c] 1
 zV[r,lb,Course,Role] choose1[Course,Role] 1 axisVdifLB[r,Course,Role] -48
 zV[r,lb,Course,Role] triangle2[Client,Course,Role,r] 1 triangle2[CertificateTemplate,Course,Role,r] 1
 zV[r,lb,Course,Role] triangle2[Course,Role,User,r] 1 triangle2[Course,Role,CourseSession,r] 1
 zV[r,lb,Course,Role] triangle2[Course,Role,Clinic,r] 1 triangle2[Course,Role,Location,r] 1
 zV[r,lb,Course,Role] triangle2[Course,Role,Therapist,r] 1 triangle2[Course,Role,Instructor,r] 1
 zV[c,lb,Course,Role] choose1[Course,Role] 1 axisVdifLB[c,Course,Role] -52
 zV[c,lb,Course,Role] triangle2[Client,Course,Role,c] 1 triangle2[CertificateTemplate,Course,Role,c] 1
 zV[c,lb,Course,Role] triangle2[Course,Role,User,c] 1 triangle2[Course,Role,CourseSession,c] 1
 zV[c,lb,Course,Role] triangle2[Course,Role,Clinic,c] 1 triangle2[Course,Role,Location,c] 1
 zV[c,lb,Course,Role] triangle2[Course,Role,Therapist,c] 1 triangle2[Course,Role,Instructor,c] 1
 zV[r,rt,Course,User] choose1[Course,User] 1 axisVdifRT[r,Course,User] -49
 zV[r,rt,Course,User] triangle1[Client,Course,User,r] 1 triangle1[CertificateTemplate,Course,User,r] 1
 zV[r,rt,Course,User] triangle1[Course,User,CourseSession,r] 1 triangle1[Course,User,Clinic,r] 1
 zV[r,rt,Course,User] triangle1[Course,User,Location,r] 1 triangle1[Course,User,Therapist,r] 1
 zV[r,rt,Course,User] triangle1[Course,User,Instructor,r] 1 triangle2[Course,Role,User,r] 1
 zV[c,rt,Course,User] choose1[Course,User] 1 axisVdifRT[c,Course,User] -53
 zV[c,rt,Course,User] triangle1[Client,Course,User,c] 1 triangle1[CertificateTemplate,Course,User,c] 1
 zV[c,rt,Course,User] triangle1[Course,User,CourseSession,c] 1 triangle1[Course,User,Clinic,c] 1
 zV[c,rt,Course,User] triangle1[Course,User,Location,c] 1 triangle1[Course,User,Therapist,c] 1
 zV[c,rt,Course,User] triangle1[Course,User,Instructor,c] 1 triangle2[Course,Role,User,c] 1
 zV[r,lb,Course,User] choose1[Course,User] 1 axisVdifLB[r,Course,User] -48
 zV[r,lb,Course,User] triangle1[Course,Role,User,r] 1 triangle2[Client,Course,User,r] 1
 zV[r,lb,Course,User] triangle2[CertificateTemplate,Course,User,r] 1 triangle2[Course,User,CourseSession,r] 1
 zV[r,lb,Course,User] triangle2[Course,User,Clinic,r] 1 triangle2[Course,User,Location,r] 1
 zV[r,lb,Course,User] triangle2[Course,User,Therapist,r] 1 triangle2[Course,User,Instructor,r] 1
 zV[c,lb,Course,User] choose1[Course,User] 1 axisVdifLB[c,Course,User] -52
 zV[c,lb,Course,User] triangle1[Course,Role,User,c] 1 triangle2[Client,Course,User,c] 1
 zV[c,lb,Course,User] triangle2[CertificateTemplate,Course,User,c] 1 triangle2[Course,User,CourseSession,c] 1
 zV[c,lb,Course,User] triangle2[Course,User,Clinic,c] 1 triangle2[Course,User,Location,c] 1
 zV[c,lb,Course,User] triangle2[Course,User,Therapist,c] 1 triangle2[Course,User,Instructor,c] 1
 zV[r,rt,Course,CourseSession] choose1[Course,CourseSession] 1 axisVdifRT[r,Course,CourseSession] -51
 zV[r,rt,Course,CourseSession] DistAxisLB1[c,Course,CourseSession] 12 triangle1[Client,Course,CourseSession,r] 1
 zV[r,rt,Course,CourseSession] triangle1[CertificateTemplate,Course,CourseSession,r] 1 triangle1[Course,CourseSession,Clinic,r] 1
 zV[r,rt,Course,CourseSession] triangle1[Course,CourseSession,Location,r] 1 triangle1[Course,CourseSession,Therapist,r] 1
 zV[r,rt,Course,CourseSession] triangle1[Course,CourseSession,Instructor,r] 1 triangle2[Course,Role,CourseSession,r] 1
 zV[r,rt,Course,CourseSession] triangle2[Course,User,CourseSession,r] 1
 zV[c,rt,Course,CourseSession] choose1[Course,CourseSession] 1 axisVdifRT[c,Course,CourseSession] -56
 zV[c,rt,Course,CourseSession] DistAxisLB1[r,Course,CourseSession] 7.5 triangle1[Client,Course,CourseSession,c] 1
 zV[c,rt,Course,CourseSession] triangle1[CertificateTemplate,Course,CourseSession,c] 1 triangle1[Course,CourseSession,Clinic,c] 1
 zV[c,rt,Course,CourseSession] triangle1[Course,CourseSession,Location,c] 1 triangle1[Course,CourseSession,Therapist,c] 1
 zV[c,rt,Course,CourseSession] triangle1[Course,CourseSession,Instructor,c] 1 triangle2[Course,Role,CourseSession,c] 1
 zV[c,rt,Course,CourseSession] triangle2[Course,User,CourseSession,c] 1
 zV[r,lb,Course,CourseSession] choose1[Course,CourseSession] 1 axisVdifLB[r,Course,CourseSession] -48
 zV[r,lb,Course,CourseSession] DistAxisLB1[c,Course,CourseSession] 12 triangle1[Course,Role,CourseSession,r] 1
 zV[r,lb,Course,CourseSession] triangle1[Course,User,CourseSession,r] 1 triangle2[Client,Course,CourseSession,r] 1
 zV[r,lb,Course,CourseSession] triangle2[CertificateTemplate,Course,CourseSession,r] 1 triangle2[Course,CourseSession,Clinic,r] 1
 zV[r,lb,Course,CourseSession] triangle2[Course,CourseSession,Location,r] 1 triangle2[Course,CourseSession,Therapist,r] 1
 zV[r,lb,Course,CourseSession] triangle2[Course,CourseSession,Instructor,r] 1
 zV[c,lb,Course,CourseSession] choose1[Course,CourseSession] 1 axisVdifLB[c,Course,CourseSession] -52
 zV[c,lb,Course,CourseSession] DistAxisLB1[r,Course,CourseSession] 7.5 triangle1[Course,Role,CourseSession,c] 1
 zV[c,lb,Course,CourseSession] triangle1[Course,User,CourseSession,c] 1 triangle2[Client,Course,CourseSession,c] 1
 zV[c,lb,Course,CourseSession] triangle2[CertificateTemplate,Course,CourseSession,c] 1 triangle2[Course,CourseSession,Clinic,c] 1
 zV[c,lb,Course,CourseSession] triangle2[Course,CourseSession,Location,c] 1 triangle2[Course,CourseSession,Therapist,c] 1
 zV[c,lb,Course,CourseSession] triangle2[Course,CourseSession,Instructor,c] 1
 zV[r,rt,Course,Clinic] choose1[Course,Clinic] 1 axisVdifRT[r,Course,Clinic] -50
 zV[r,rt,Course,Clinic] triangle1[Client,Course,Clinic,r] 1 triangle1[CertificateTemplate,Course,Clinic,r] 1
 zV[r,rt,Course,Clinic] triangle1[Course,Clinic,Location,r] 1 triangle1[Course,Clinic,Therapist,r] 1
 zV[r,rt,Course,Clinic] triangle1[Course,Clinic,Instructor,r] 1 triangle2[Course,Role,Clinic,r] 1
 zV[r,rt,Course,Clinic] triangle2[Course,User,Clinic,r] 1 triangle2[Course,CourseSession,Clinic,r] 1
 zV[c,rt,Course,Clinic] choose1[Course,Clinic] 1 axisVdifRT[c,Course,Clinic] -53
 zV[c,rt,Course,Clinic] triangle1[Client,Course,Clinic,c] 1 triangle1[CertificateTemplate,Course,Clinic,c] 1
 zV[c,rt,Course,Clinic] triangle1[Course,Clinic,Location,c] 1 triangle1[Course,Clinic,Therapist,c] 1
 zV[c,rt,Course,Clinic] triangle1[Course,Clinic,Instructor,c] 1 triangle2[Course,Role,Clinic,c] 1
 zV[c,rt,Course,Clinic] triangle2[Course,User,Clinic,c] 1 triangle2[Course,CourseSession,Clinic,c] 1
 zV[r,lb,Course,Clinic] choose1[Course,Clinic] 1 axisVdifLB[r,Course,Clinic] -48
 zV[r,lb,Course,Clinic] triangle1[Course,Role,Clinic,r] 1 triangle1[Course,User,Clinic,r] 1
 zV[r,lb,Course,Clinic] triangle1[Course,CourseSession,Clinic,r] 1 triangle2[Client,Course,Clinic,r] 1
 zV[r,lb,Course,Clinic] triangle2[CertificateTemplate,Course,Clinic,r] 1 triangle2[Course,Clinic,Location,r] 1
 zV[r,lb,Course,Clinic] triangle2[Course,Clinic,Therapist,r] 1 triangle2[Course,Clinic,Instructor,r] 1
 zV[c,lb,Course,Clinic] choose1[Course,Clinic] 1 axisVdifLB[c,Course,Clinic] -52
 zV[c,lb,Course,Clinic] triangle1[Course,Role,Clinic,c] 1 triangle1[Course,User,Clinic,c] 1
 zV[c,lb,Course,Clinic] triangle1[Course,CourseSession,Clinic,c] 1 triangle2[Client,Course,Clinic,c] 1
 zV[c,lb,Course,Clinic] triangle2[CertificateTemplate,Course,Clinic,c] 1 triangle2[Course,Clinic,Location,c] 1
 zV[c,lb,Course,Clinic] triangle2[Course,Clinic,Therapist,c] 1 triangle2[Course,Clinic,Instructor,c] 1
 zV[r,rt,Course,Location] choose1[Course,Location] 1 axisVdifRT[r,Course,Location] -51
 zV[r,rt,Course,Location] triangle1[Client,Course,Location,r] 1 triangle1[CertificateTemplate,Course,Location,r] 1
 zV[r,rt,Course,Location] triangle1[Course,Location,Therapist,r] 1 triangle1[Course,Location,Instructor,r] 1
 zV[r,rt,Course,Location] triangle2[Course,Role,Location,r] 1 triangle2[Course,User,Location,r] 1
 zV[r,rt,Course,Location] triangle2[Course,CourseSession,Location,r] 1 triangle2[Course,Clinic,Location,r] 1
 zV[c,rt,Course,Location] choose1[Course,Location] 1 axisVdifRT[c,Course,Location] -53
 zV[c,rt,Course,Location] triangle1[Client,Course,Location,c] 1 triangle1[CertificateTemplate,Course,Location,c] 1
 zV[c,rt,Course,Location] triangle1[Course,Location,Therapist,c] 1 triangle1[Course,Location,Instructor,c] 1
 zV[c,rt,Course,Location] triangle2[Course,Role,Location,c] 1 triangle2[Course,User,Location,c] 1
 zV[c,rt,Course,Location] triangle2[Course,CourseSession,Location,c] 1 triangle2[Course,Clinic,Location,c] 1
 zV[r,lb,Course,Location] choose1[Course,Location] 1 axisVdifLB[r,Course,Location] -48
 zV[r,lb,Course,Location] triangle1[Course,Role,Location,r] 1 triangle1[Course,User,Location,r] 1
 zV[r,lb,Course,Location] triangle1[Course,CourseSession,Location,r] 1 triangle1[Course,Clinic,Location,r] 1
 zV[r,lb,Course,Location] triangle2[Client,Course,Location,r] 1 triangle2[CertificateTemplate,Course,Location,r] 1
 zV[r,lb,Course,Location] triangle2[Course,Location,Therapist,r] 1 triangle2[Course,Location,Instructor,r] 1
 zV[c,lb,Course,Location] choose1[Course,Location] 1 axisVdifLB[c,Course,Location] -52
 zV[c,lb,Course,Location] triangle1[Course,Role,Location,c] 1 triangle1[Course,User,Location,c] 1
 zV[c,lb,Course,Location] triangle1[Course,CourseSession,Location,c] 1 triangle1[Course,Clinic,Location,c] 1
 zV[c,lb,Course,Location] triangle2[Client,Course,Location,c] 1 triangle2[CertificateTemplate,Course,Location,c] 1
 zV[c,lb,Course,Location] triangle2[Course,Location,Therapist,c] 1 triangle2[Course,Location,Instructor,c] 1
 zV[r,rt,Course,Therapist] choose1[Course,Therapist] 1 axisVdifRT[r,Course,Therapist] -52
 zV[r,rt,Course,Therapist] triangle1[Client,Course,Therapist,r] 1 triangle1[CertificateTemplate,Course,Therapist,r] 1
 zV[r,rt,Course,Therapist] triangle1[Course,Therapist,Instructor,r] 1 triangle2[Course,Role,Therapist,r] 1
 zV[r,rt,Course,Therapist] triangle2[Course,User,Therapist,r] 1 triangle2[Course,CourseSession,Therapist,r] 1
 zV[r,rt,Course,Therapist] triangle2[Course,Clinic,Therapist,r] 1 triangle2[Course,Location,Therapist,r] 1
 zV[c,rt,Course,Therapist] choose1[Course,Therapist] 1 axisVdifRT[c,Course,Therapist] -53
 zV[c,rt,Course,Therapist] triangle1[Client,Course,Therapist,c] 1 triangle1[CertificateTemplate,Course,Therapist,c] 1
 zV[c,rt,Course,Therapist] triangle1[Course,Therapist,Instructor,c] 1 triangle2[Course,Role,Therapist,c] 1
 zV[c,rt,Course,Therapist] triangle2[Course,User,Therapist,c] 1 triangle2[Course,CourseSession,Therapist,c] 1
 zV[c,rt,Course,Therapist] triangle2[Course,Clinic,Therapist,c] 1 triangle2[Course,Location,Therapist,c] 1
 zV[r,lb,Course,Therapist] choose1[Course,Therapist] 1 axisVdifLB[r,Course,Therapist] -48
 zV[r,lb,Course,Therapist] triangle1[Course,Role,Therapist,r] 1 triangle1[Course,User,Therapist,r] 1
 zV[r,lb,Course,Therapist] triangle1[Course,CourseSession,Therapist,r] 1 triangle1[Course,Clinic,Therapist,r] 1
 zV[r,lb,Course,Therapist] triangle1[Course,Location,Therapist,r] 1 triangle2[Client,Course,Therapist,r] 1
 zV[r,lb,Course,Therapist] triangle2[CertificateTemplate,Course,Therapist,r] 1 triangle2[Course,Therapist,Instructor,r] 1
 zV[c,lb,Course,Therapist] choose1[Course,Therapist] 1 axisVdifLB[c,Course,Therapist] -52
 zV[c,lb,Course,Therapist] triangle1[Course,Role,Therapist,c] 1 triangle1[Course,User,Therapist,c] 1
 zV[c,lb,Course,Therapist] triangle1[Course,CourseSession,Therapist,c] 1 triangle1[Course,Clinic,Therapist,c] 1
 zV[c,lb,Course,Therapist] triangle1[Course,Location,Therapist,c] 1 triangle2[Client,Course,Therapist,c] 1
 zV[c,lb,Course,Therapist] triangle2[CertificateTemplate,Course,Therapist,c] 1 triangle2[Course,Therapist,Instructor,c] 1
 zV[r,rt,Course,Instructor] choose1[Course,Instructor] 1 axisVdifRT[r,Course,Instructor] -52
 zV[r,rt,Course,Instructor] triangle1[Client,Course,Instructor,r] 1 triangle1[CertificateTemplate,Course,Instructor,r] 1
 zV[r,rt,Course,Instructor] triangle2[Course,Role,Instructor,r] 1 triangle2[Course,User,Instructor,r] 1
 zV[r,rt,Course,Instructor] triangle2[Course,CourseSession,Instructor,r] 1 triangle2[Course,Clinic,Instructor,r] 1
 zV[r,rt,Course,Instructor] triangle2[Course,Location,Instructor,r] 1 triangle2[Course,Therapist,Instructor,r] 1
 zV[c,rt,Course,Instructor] choose1[Course,Instructor] 1 axisVdifRT[c,Course,Instructor] -54
 zV[c,rt,Course,Instructor] triangle1[Client,Course,Instructor,c] 1 triangle1[CertificateTemplate,Course,Instructor,c] 1
 zV[c,rt,Course,Instructor] triangle2[Course,Role,Instructor,c] 1 triangle2[Course,User,Instructor,c] 1
 zV[c,rt,Course,Instructor] triangle2[Course,CourseSession,Instructor,c] 1 triangle2[Course,Clinic,Instructor,c] 1
 zV[c,rt,Course,Instructor] triangle2[Course,Location,Instructor,c] 1 triangle2[Course,Therapist,Instructor,c] 1
 zV[r,lb,Course,Instructor] choose1[Course,Instructor] 1 axisVdifLB[r,Course,Instructor] -48
 zV[r,lb,Course,Instructor] triangle1[Course,Role,Instructor,r] 1 triangle1[Course,User,Instructor,r] 1
 zV[r,lb,Course,Instructor] triangle1[Course,CourseSession,Instructor,r] 1 triangle1[Course,Clinic,Instructor,r] 1
 zV[r,lb,Course,Instructor] triangle1[Course,Location,Instructor,r] 1 triangle1[Course,Therapist,Instructor,r] 1
 zV[r,lb,Course,Instructor] triangle2[Client,Course,Instructor,r] 1 triangle2[CertificateTemplate,Course,Instructor,r] 1
 zV[c,lb,Course,Instructor] choose1[Course,Instructor] 1 axisVdifLB[c,Course,Instructor] -52
 zV[c,lb,Course,Instructor] triangle1[Course,Role,Instructor,c] 1 triangle1[Course,User,Instructor,c] 1
 zV[c,lb,Course,Instructor] triangle1[Course,CourseSession,Instructor,c] 1 triangle1[Course,Clinic,Instructor,c] 1
 zV[c,lb,Course,Instructor] triangle1[Course,Location,Instructor,c] 1 triangle1[Course,Therapist,Instructor,c] 1
 zV[c,lb,Course,Instructor] triangle2[Client,Course,Instructor,c] 1 triangle2[CertificateTemplate,Course,Instructor,c] 1
 zV[r,rt,Role,User] choose1[Role,User] 1 axisVdifRT[r,Role,User] -49
 zV[r,rt,Role,User] DistAxisLB1[c,Role,User] 10.5 triangle1[Client,Role,User,r] 1
 zV[r,rt,Role,User] triangle1[CertificateTemplate,Role,User,r] 1 triangle1[Course,Role,User,r] 1
 zV[r,rt,Role,User] triangle1[Role,User,CourseSession,r] 1 triangle1[Role,User,Clinic,r] 1
 zV[r,rt,Role,User] triangle1[Role,User,Location,r] 1 triangle1[Role,User,Therapist,r] 1
 zV[r,rt,Role,User] triangle1[Role,User,Instructor,r] 1
 zV[c,rt,Role,User] choose1[Role,User] 1 axisVdifRT[c,Role,User] -53
 zV[c,rt,Role,User] DistAxisLB1[r,Role,User] 6.5 triangle1[Client,Role,User,c] 1
 zV[c,rt,Role,User] triangle1[CertificateTemplate,Role,User,c] 1 triangle1[Course,Role,User,c] 1
 zV[c,rt,Role,User] triangle1[Role,User,CourseSession,c] 1 triangle1[Role,User,Clinic,c] 1
 zV[c,rt,Role,User] triangle1[Role,User,Location,c] 1 triangle1[Role,User,Therapist,c] 1
 zV[c,rt,Role,User] triangle1[Role,User,Instructor,c] 1
 zV[r,lb,Role,User] choose1[Role,User] 1 axisVdifLB[r,Role,User] -48
 zV[r,lb,Role,User] DistAxisLB1[c,Role,User] 10.5 triangle2[Client,Role,User,r] 1
 zV[r,lb,Role,User] triangle2[CertificateTemplate,Role,User,r] 1 triangle2[Course,Role,User,r] 1
 zV[r,lb,Role,User] triangle2[Role,User,CourseSession,r] 1 triangle2[Role,User,Clinic,r] 1
 zV[r,lb,Role,User] triangle2[Role,User,Location,r] 1 triangle2[Role,User,Therapist,r] 1
 zV[r,lb,Role,User] triangle2[Role,User,Instructor,r] 1
 zV[c,lb,Role,User] choose1[Role,User] 1 axisVdifLB[c,Role,User] -52
 zV[c,lb,Role,User] DistAxisLB1[r,Role,User] 6.5 triangle2[Client,Role,User,c] 1
 zV[c,lb,Role,User] triangle2[CertificateTemplate,Role,User,c] 1 triangle2[Course,Role,User,c] 1
 zV[c,lb,Role,User] triangle2[Role,User,CourseSession,c] 1 triangle2[Role,User,Clinic,c] 1
 zV[c,lb,Role,User] triangle2[Role,User,Location,c] 1 triangle2[Role,User,Therapist,c] 1
 zV[c,lb,Role,User] triangle2[Role,User,Instructor,c] 1
 zV[r,rt,Role,CourseSession] choose1[Role,CourseSession] 1 axisVdifRT[r,Role,CourseSession] -51
 zV[r,rt,Role,CourseSession] triangle1[Client,Role,CourseSession,r] 1 triangle1[CertificateTemplate,Role,CourseSession,r] 1
 zV[r,rt,Role,CourseSession] triangle1[Course,Role,CourseSession,r] 1 triangle1[Role,CourseSession,Clinic,r] 1
 zV[r,rt,Role,CourseSession] triangle1[Role,CourseSession,Location,r] 1 triangle1[Role,CourseSession,Therapist,r] 1
 zV[r,rt,Role,CourseSession] triangle1[Role,CourseSession,Instructor,r] 1 triangle2[Role,User,CourseSession,r] 1
 zV[c,rt,Role,CourseSession] choose1[Role,CourseSession] 1 axisVdifRT[c,Role,CourseSession] -56
 zV[c,rt,Role,CourseSession] triangle1[Client,Role,CourseSession,c] 1 triangle1[CertificateTemplate,Role,CourseSession,c] 1
 zV[c,rt,Role,CourseSession] triangle1[Course,Role,CourseSession,c] 1 triangle1[Role,CourseSession,Clinic,c] 1
 zV[c,rt,Role,CourseSession] triangle1[Role,CourseSession,Location,c] 1 triangle1[Role,CourseSession,Therapist,c] 1
 zV[c,rt,Role,CourseSession] triangle1[Role,CourseSession,Instructor,c] 1 triangle2[Role,User,CourseSession,c] 1
 zV[r,lb,Role,CourseSession] choose1[Role,CourseSession] 1 axisVdifLB[r,Role,CourseSession] -48
 zV[r,lb,Role,CourseSession] triangle1[Role,User,CourseSession,r] 1 triangle2[Client,Role,CourseSession,r] 1
 zV[r,lb,Role,CourseSession] triangle2[CertificateTemplate,Role,CourseSession,r] 1 triangle2[Course,Role,CourseSession,r] 1
 zV[r,lb,Role,CourseSession] triangle2[Role,CourseSession,Clinic,r] 1 triangle2[Role,CourseSession,Location,r] 1
 zV[r,lb,Role,CourseSession] triangle2[Role,CourseSession,Therapist,r] 1 triangle2[Role,CourseSession,Instructor,r] 1
 zV[c,lb,Role,CourseSession] choose1[Role,CourseSession] 1 axisVdifLB[c,Role,CourseSession] -52
 zV[c,lb,Role,CourseSession] triangle1[Role,User,CourseSession,c] 1 triangle2[Client,Role,CourseSession,c] 1
 zV[c,lb,Role,CourseSession] triangle2[CertificateTemplate,Role,CourseSession,c] 1 triangle2[Course,Role,CourseSession,c] 1
 zV[c,lb,Role,CourseSession] triangle2[Role,CourseSession,Clinic,c] 1 triangle2[Role,CourseSession,Location,c] 1
 zV[c,lb,Role,CourseSession] triangle2[Role,CourseSession,Therapist,c] 1 triangle2[Role,CourseSession,Instructor,c] 1
 zV[r,rt,Role,Clinic] choose1[Role,Clinic] 1 axisVdifRT[r,Role,Clinic] -50
 zV[r,rt,Role,Clinic] triangle1[Client,Role,Clinic,r] 1 triangle1[CertificateTemplate,Role,Clinic,r] 1
 zV[r,rt,Role,Clinic] triangle1[Course,Role,Clinic,r] 1 triangle1[Role,Clinic,Location,r] 1
 zV[r,rt,Role,Clinic] triangle1[Role,Clinic,Therapist,r] 1 triangle1[Role,Clinic,Instructor,r] 1
 zV[r,rt,Role,Clinic] triangle2[Role,User,Clinic,r] 1 triangle2[Role,CourseSession,Clinic,r] 1
 zV[c,rt,Role,Clinic] choose1[Role,Clinic] 1 axisVdifRT[c,Role,Clinic] -53
 zV[c,rt,Role,Clinic] triangle1[Client,Role,Clinic,c] 1 triangle1[CertificateTemplate,Role,Clinic,c] 1
 zV[c,rt,Role,Clinic] triangle1[Course,Role,Clinic,c] 1 triangle1[Role,Clinic,Location,c] 1
 zV[c,rt,Role,Clinic] triangle1[Role,Clinic,Therapist,c] 1 triangle1[Role,Clinic,Instructor,c] 1
 zV[c,rt,Role,Clinic] triangle2[Role,User,Clinic,c] 1 triangle2[Role,CourseSession,Clinic,c] 1
 zV[r,lb,Role,Clinic] choose1[Role,Clinic] 1 axisVdifLB[r,Role,Clinic] -48
 zV[r,lb,Role,Clinic] triangle1[Role,User,Clinic,r] 1 triangle1[Role,CourseSession,Clinic,r] 1
 zV[r,lb,Role,Clinic] triangle2[Client,Role,Clinic,r] 1 triangle2[CertificateTemplate,Role,Clinic,r] 1
 zV[r,lb,Role,Clinic] triangle2[Course,Role,Clinic,r] 1 triangle2[Role,Clinic,Location,r] 1
 zV[r,lb,Role,Clinic] triangle2[Role,Clinic,Therapist,r] 1 triangle2[Role,Clinic,Instructor,r] 1
 zV[c,lb,Role,Clinic] choose1[Role,Clinic] 1 axisVdifLB[c,Role,Clinic] -52
 zV[c,lb,Role,Clinic] triangle1[Role,User,Clinic,c] 1 triangle1[Role,CourseSession,Clinic,c] 1
 zV[c,lb,Role,Clinic] triangle2[Client,Role,Clinic,c] 1 triangle2[CertificateTemplate,Role,Clinic,c] 1
 zV[c,lb,Role,Clinic] triangle2[Course,Role,Clinic,c] 1 triangle2[Role,Clinic,Location,c] 1
 zV[c,lb,Role,Clinic] triangle2[Role,Clinic,Therapist,c] 1 triangle2[Role,Clinic,Instructor,c] 1
 zV[r,rt,Role,Location] choose1[Role,Location] 1 axisVdifRT[r,Role,Location] -51
 zV[r,rt,Role,Location] triangle1[Client,Role,Location,r] 1 triangle1[CertificateTemplate,Role,Location,r] 1
 zV[r,rt,Role,Location] triangle1[Course,Role,Location,r] 1 triangle1[Role,Location,Therapist,r] 1
 zV[r,rt,Role,Location] triangle1[Role,Location,Instructor,r] 1 triangle2[Role,User,Location,r] 1
 zV[r,rt,Role,Location] triangle2[Role,CourseSession,Location,r] 1 triangle2[Role,Clinic,Location,r] 1
 zV[c,rt,Role,Location] choose1[Role,Location] 1 axisVdifRT[c,Role,Location] -53
 zV[c,rt,Role,Location] triangle1[Client,Role,Location,c] 1 triangle1[CertificateTemplate,Role,Location,c] 1
 zV[c,rt,Role,Location] triangle1[Course,Role,Location,c] 1 triangle1[Role,Location,Therapist,c] 1
 zV[c,rt,Role,Location] triangle1[Role,Location,Instructor,c] 1 triangle2[Role,User,Location,c] 1
 zV[c,rt,Role,Location] triangle2[Role,CourseSession,Location,c] 1 triangle2[Role,Clinic,Location,c] 1
 zV[r,lb,Role,Location] choose1[Role,Location] 1 axisVdifLB[r,Role,Location] -48
 zV[r,lb,Role,Location] triangle1[Role,User,Location,r] 1 triangle1[Role,CourseSession,Location,r] 1
 zV[r,lb,Role,Location] triangle1[Role,Clinic,Location,r] 1 triangle2[Client,Role,Location,r] 1
 zV[r,lb,Role,Location] triangle2[CertificateTemplate,Role,Location,r] 1 triangle2[Course,Role,Location,r] 1
 zV[r,lb,Role,Location] triangle2[Role,Location,Therapist,r] 1 triangle2[Role,Location,Instructor,r] 1
 zV[c,lb,Role,Location] choose1[Role,Location] 1 axisVdifLB[c,Role,Location] -52
 zV[c,lb,Role,Location] triangle1[Role,User,Location,c] 1 triangle1[Role,CourseSession,Location,c] 1
 zV[c,lb,Role,Location] triangle1[Role,Clinic,Location,c] 1 triangle2[Client,Role,Location,c] 1
 zV[c,lb,Role,Location] triangle2[CertificateTemplate,Role,Location,c] 1 triangle2[Course,Role,Location,c] 1
 zV[c,lb,Role,Location] triangle2[Role,Location,Therapist,c] 1 triangle2[Role,Location,Instructor,c] 1
 zV[r,rt,Role,Therapist] choose1[Role,Therapist] 1 axisVdifRT[r,Role,Therapist] -52
 zV[r,rt,Role,Therapist] triangle1[Client,Role,Therapist,r] 1 triangle1[CertificateTemplate,Role,Therapist,r] 1
 zV[r,rt,Role,Therapist] triangle1[Course,Role,Therapist,r] 1 triangle1[Role,Therapist,Instructor,r] 1
 zV[r,rt,Role,Therapist] triangle2[Role,User,Therapist,r] 1 triangle2[Role,CourseSession,Therapist,r] 1
 zV[r,rt,Role,Therapist] triangle2[Role,Clinic,Therapist,r] 1 triangle2[Role,Location,Therapist,r] 1
 zV[c,rt,Role,Therapist] choose1[Role,Therapist] 1 axisVdifRT[c,Role,Therapist] -53
 zV[c,rt,Role,Therapist] triangle1[Client,Role,Therapist,c] 1 triangle1[CertificateTemplate,Role,Therapist,c] 1
 zV[c,rt,Role,Therapist] triangle1[Course,Role,Therapist,c] 1 triangle1[Role,Therapist,Instructor,c] 1
 zV[c,rt,Role,Therapist] triangle2[Role,User,Therapist,c] 1 triangle2[Role,CourseSession,Therapist,c] 1
 zV[c,rt,Role,Therapist] triangle2[Role,Clinic,Therapist,c] 1 triangle2[Role,Location,Therapist,c] 1
 zV[r,lb,Role,Therapist] choose1[Role,Therapist] 1 axisVdifLB[r,Role,Therapist] -48
 zV[r,lb,Role,Therapist] triangle1[Role,User,Therapist,r] 1 triangle1[Role,CourseSession,Therapist,r] 1
 zV[r,lb,Role,Therapist] triangle1[Role,Clinic,Therapist,r] 1 triangle1[Role,Location,Therapist,r] 1
 zV[r,lb,Role,Therapist] triangle2[Client,Role,Therapist,r] 1 triangle2[CertificateTemplate,Role,Therapist,r] 1
 zV[r,lb,Role,Therapist] triangle2[Course,Role,Therapist,r] 1 triangle2[Role,Therapist,Instructor,r] 1
 zV[c,lb,Role,Therapist] choose1[Role,Therapist] 1 axisVdifLB[c,Role,Therapist] -52
 zV[c,lb,Role,Therapist] triangle1[Role,User,Therapist,c] 1 triangle1[Role,CourseSession,Therapist,c] 1
 zV[c,lb,Role,Therapist] triangle1[Role,Clinic,Therapist,c] 1 triangle1[Role,Location,Therapist,c] 1
 zV[c,lb,Role,Therapist] triangle2[Client,Role,Therapist,c] 1 triangle2[CertificateTemplate,Role,Therapist,c] 1
 zV[c,lb,Role,Therapist] triangle2[Course,Role,Therapist,c] 1 triangle2[Role,Therapist,Instructor,c] 1
 zV[r,rt,Role,Instructor] choose1[Role,Instructor] 1 axisVdifRT[r,Role,Instructor] -52
 zV[r,rt,Role,Instructor] triangle1[Client,Role,Instructor,r] 1 triangle1[CertificateTemplate,Role,Instructor,r] 1
 zV[r,rt,Role,Instructor] triangle1[Course,Role,Instructor,r] 1 triangle2[Role,User,Instructor,r] 1
 zV[r,rt,Role,Instructor] triangle2[Role,CourseSession,Instructor,r] 1 triangle2[Role,Clinic,Instructor,r] 1
 zV[r,rt,Role,Instructor] triangle2[Role,Location,Instructor,r] 1 triangle2[Role,Therapist,Instructor,r] 1
 zV[c,rt,Role,Instructor] choose1[Role,Instructor] 1 axisVdifRT[c,Role,Instructor] -54
 zV[c,rt,Role,Instructor] triangle1[Client,Role,Instructor,c] 1 triangle1[CertificateTemplate,Role,Instructor,c] 1
 zV[c,rt,Role,Instructor] triangle1[Course,Role,Instructor,c] 1 triangle2[Role,User,Instructor,c] 1
 zV[c,rt,Role,Instructor] triangle2[Role,CourseSession,Instructor,c] 1 triangle2[Role,Clinic,Instructor,c] 1
 zV[c,rt,Role,Instructor] triangle2[Role,Location,Instructor,c] 1 triangle2[Role,Therapist,Instructor,c] 1
 zV[r,lb,Role,Instructor] choose1[Role,Instructor] 1 axisVdifLB[r,Role,Instructor] -48
 zV[r,lb,Role,Instructor] triangle1[Role,User,Instructor,r] 1 triangle1[Role,CourseSession,Instructor,r] 1
 zV[r,lb,Role,Instructor] triangle1[Role,Clinic,Instructor,r] 1 triangle1[Role,Location,Instructor,r] 1
 zV[r,lb,Role,Instructor] triangle1[Role,Therapist,Instructor,r] 1 triangle2[Client,Role,Instructor,r] 1
 zV[r,lb,Role,Instructor] triangle2[CertificateTemplate,Role,Instructor,r] 1 triangle2[Course,Role,Instructor,r] 1
 zV[c,lb,Role,Instructor] choose1[Role,Instructor] 1 axisVdifLB[c,Role,Instructor] -52
 zV[c,lb,Role,Instructor] triangle1[Role,User,Instructor,c] 1 triangle1[Role,CourseSession,Instructor,c] 1
 zV[c,lb,Role,Instructor] triangle1[Role,Clinic,Instructor,c] 1 triangle1[Role,Location,Instructor,c] 1
 zV[c,lb,Role,Instructor] triangle1[Role,Therapist,Instructor,c] 1 triangle2[Client,Role,Instructor,c] 1
 zV[c,lb,Role,Instructor] triangle2[CertificateTemplate,Role,Instructor,c] 1 triangle2[Course,Role,Instructor,c] 1
 zV[r,rt,User,CourseSession] choose1[User,CourseSession] 1 axisVdifRT[r,User,CourseSession] -51
 zV[r,rt,User,CourseSession] triangle1[Client,User,CourseSession,r] 1 triangle1[CertificateTemplate,User,CourseSession,r] 1
 zV[r,rt,User,CourseSession] triangle1[Course,User,CourseSession,r] 1 triangle1[Role,User,CourseSession,r] 1
 zV[r,rt,User,CourseSession] triangle1[User,CourseSession,Clinic,r] 1 triangle1[User,CourseSession,Location,r] 1
 zV[r,rt,User,CourseSession] triangle1[User,CourseSession,Therapist,r] 1 triangle1[User,CourseSession,Instructor,r] 1
 zV[c,rt,User,CourseSession] choose1[User,CourseSession] 1 axisVdifRT[c,User,CourseSession] -56
 zV[c,rt,User,CourseSession] triangle1[Client,User,CourseSession,c] 1 triangle1[CertificateTemplate,User,CourseSession,c] 1
 zV[c,rt,User,CourseSession] triangle1[Course,User,CourseSession,c] 1 triangle1[Role,User,CourseSession,c] 1
 zV[c,rt,User,CourseSession] triangle1[User,CourseSession,Clinic,c] 1 triangle1[User,CourseSession,Location,c] 1
 zV[c,rt,User,CourseSession] triangle1[User,CourseSession,Therapist,c] 1 triangle1[User,CourseSession,Instructor,c] 1
 zV[r,lb,User,CourseSession] choose1[User,CourseSession] 1 axisVdifLB[r,User,CourseSession] -49
 zV[r,lb,User,CourseSession] triangle2[Client,User,CourseSession,r] 1 triangle2[CertificateTemplate,User,CourseSession,r] 1
 zV[r,lb,User,CourseSession] triangle2[Course,User,CourseSession,r] 1 triangle2[Role,User,CourseSession,r] 1
 zV[r,lb,User,CourseSession] triangle2[User,CourseSession,Clinic,r] 1 triangle2[User,CourseSession,Location,r] 1
 zV[r,lb,User,CourseSession] triangle2[User,CourseSession,Therapist,r] 1 triangle2[User,CourseSession,Instructor,r] 1
 zV[c,lb,User,CourseSession] choose1[User,CourseSession] 1 axisVdifLB[c,User,CourseSession] -53
 zV[c,lb,User,CourseSession] triangle2[Client,User,CourseSession,c] 1 triangle2[CertificateTemplate,User,CourseSession,c] 1
 zV[c,lb,User,CourseSession] triangle2[Course,User,CourseSession,c] 1 triangle2[Role,User,CourseSession,c] 1
 zV[c,lb,User,CourseSession] triangle2[User,CourseSession,Clinic,c] 1 triangle2[User,CourseSession,Location,c] 1
 zV[c,lb,User,CourseSession] triangle2[User,CourseSession,Therapist,c] 1 triangle2[User,CourseSession,Instructor,c] 1
 zV[r,rt,User,Clinic] choose1[User,Clinic] 1 axisVdifRT[r,User,Clinic] -50
 zV[r,rt,User,Clinic] triangle1[Client,User,Clinic,r] 1 triangle1[CertificateTemplate,User,Clinic,r] 1
 zV[r,rt,User,Clinic] triangle1[Course,User,Clinic,r] 1 triangle1[Role,User,Clinic,r] 1
 zV[r,rt,User,Clinic] triangle1[User,Clinic,Location,r] 1 triangle1[User,Clinic,Therapist,r] 1
 zV[r,rt,User,Clinic] triangle1[User,Clinic,Instructor,r] 1 triangle2[User,CourseSession,Clinic,r] 1
 zV[c,rt,User,Clinic] choose1[User,Clinic] 1 axisVdifRT[c,User,Clinic] -53
 zV[c,rt,User,Clinic] triangle1[Client,User,Clinic,c] 1 triangle1[CertificateTemplate,User,Clinic,c] 1
 zV[c,rt,User,Clinic] triangle1[Course,User,Clinic,c] 1 triangle1[Role,User,Clinic,c] 1
 zV[c,rt,User,Clinic] triangle1[User,Clinic,Location,c] 1 triangle1[User,Clinic,Therapist,c] 1
 zV[c,rt,User,Clinic] triangle1[User,Clinic,Instructor,c] 1 triangle2[User,CourseSession,Clinic,c] 1
 zV[r,lb,User,Clinic] choose1[User,Clinic] 1 axisVdifLB[r,User,Clinic] -49
 zV[r,lb,User,Clinic] triangle1[User,CourseSession,Clinic,r] 1 triangle2[Client,User,Clinic,r] 1
 zV[r,lb,User,Clinic] triangle2[CertificateTemplate,User,Clinic,r] 1 triangle2[Course,User,Clinic,r] 1
 zV[r,lb,User,Clinic] triangle2[Role,User,Clinic,r] 1 triangle2[User,Clinic,Location,r] 1
 zV[r,lb,User,Clinic] triangle2[User,Clinic,Therapist,r] 1 triangle2[User,Clinic,Instructor,r] 1
 zV[c,lb,User,Clinic] choose1[User,Clinic] 1 axisVdifLB[c,User,Clinic] -53
 zV[c,lb,User,Clinic] triangle1[User,CourseSession,Clinic,c] 1 triangle2[Client,User,Clinic,c] 1
 zV[c,lb,User,Clinic] triangle2[CertificateTemplate,User,Clinic,c] 1 triangle2[Course,User,Clinic,c] 1
 zV[c,lb,User,Clinic] triangle2[Role,User,Clinic,c] 1 triangle2[User,Clinic,Location,c] 1
 zV[c,lb,User,Clinic] triangle2[User,Clinic,Therapist,c] 1 triangle2[User,Clinic,Instructor,c] 1
 zV[r,rt,User,Location] choose1[User,Location] 1 axisVdifRT[r,User,Location] -51
 zV[r,rt,User,Location] triangle1[Client,User,Location,r] 1 triangle1[CertificateTemplate,User,Location,r] 1
 zV[r,rt,User,Location] triangle1[Course,User,Location,r] 1 triangle1[Role,User,Location,r] 1
 zV[r,rt,User,Location] triangle1[User,Location,Therapist,r] 1 triangle1[User,Location,Instructor,r] 1
 zV[r,rt,User,Location] triangle2[User,CourseSession,Location,r] 1 triangle2[User,Clinic,Location,r] 1
 zV[c,rt,User,Location] choose1[User,Location] 1 axisVdifRT[c,User,Location] -53
 zV[c,rt,User,Location] triangle1[Client,User,Location,c] 1 triangle1[CertificateTemplate,User,Location,c] 1
 zV[c,rt,User,Location] triangle1[Course,User,Location,c] 1 triangle1[Role,User,Location,c] 1
 zV[c,rt,User,Location] triangle1[User,Location,Therapist,c] 1 triangle1[User,Location,Instructor,c] 1
 zV[c,rt,User,Location] triangle2[User,CourseSession,Location,c] 1 triangle2[User,Clinic,Location,c] 1
 zV[r,lb,User,Location] choose1[User,Location] 1 axisVdifLB[r,User,Location] -49
 zV[r,lb,User,Location] triangle1[User,CourseSession,Location,r] 1 triangle1[User,Clinic,Location,r] 1
 zV[r,lb,User,Location] triangle2[Client,User,Location,r] 1 triangle2[CertificateTemplate,User,Location,r] 1
 zV[r,lb,User,Location] triangle2[Course,User,Location,r] 1 triangle2[Role,User,Location,r] 1
 zV[r,lb,User,Location] triangle2[User,Location,Therapist,r] 1 triangle2[User,Location,Instructor,r] 1
 zV[c,lb,User,Location] choose1[User,Location] 1 axisVdifLB[c,User,Location] -53
 zV[c,lb,User,Location] triangle1[User,CourseSession,Location,c] 1 triangle1[User,Clinic,Location,c] 1
 zV[c,lb,User,Location] triangle2[Client,User,Location,c] 1 triangle2[CertificateTemplate,User,Location,c] 1
 zV[c,lb,User,Location] triangle2[Course,User,Location,c] 1 triangle2[Role,User,Location,c] 1
 zV[c,lb,User,Location] triangle2[User,Location,Therapist,c] 1 triangle2[User,Location,Instructor,c] 1
 zV[r,rt,User,Therapist] choose1[User,Therapist] 1 axisVdifRT[r,User,Therapist] -52
 zV[r,rt,User,Therapist] DistAxisLB1[c,User,Therapist] 11 triangle1[Client,User,Therapist,r] 1
 zV[r,rt,User,Therapist] triangle1[CertificateTemplate,User,Therapist,r] 1 triangle1[Course,User,Therapist,r] 1
 zV[r,rt,User,Therapist] triangle1[Role,User,Therapist,r] 1 triangle1[User,Therapist,Instructor,r] 1
 zV[r,rt,User,Therapist] triangle2[User,CourseSession,Therapist,r] 1 triangle2[User,Clinic,Therapist,r] 1
 zV[r,rt,User,Therapist] triangle2[User,Location,Therapist,r] 1
 zV[c,rt,User,Therapist] choose1[User,Therapist] 1 axisVdifRT[c,User,Therapist] -53
 zV[c,rt,User,Therapist] DistAxisLB1[r,User,Therapist] 8.5 triangle1[Client,User,Therapist,c] 1
 zV[c,rt,User,Therapist] triangle1[CertificateTemplate,User,Therapist,c] 1 triangle1[Course,User,Therapist,c] 1
 zV[c,rt,User,Therapist] triangle1[Role,User,Therapist,c] 1 triangle1[User,Therapist,Instructor,c] 1
 zV[c,rt,User,Therapist] triangle2[User,CourseSession,Therapist,c] 1 triangle2[User,Clinic,Therapist,c] 1
 zV[c,rt,User,Therapist] triangle2[User,Location,Therapist,c] 1
 zV[r,lb,User,Therapist] choose1[User,Therapist] 1 axisVdifLB[r,User,Therapist] -49
 zV[r,lb,User,Therapist] DistAxisLB1[c,User,Therapist] 11 triangle1[User,CourseSession,Therapist,r] 1
 zV[r,lb,User,Therapist] triangle1[User,Clinic,Therapist,r] 1 triangle1[User,Location,Therapist,r] 1
 zV[r,lb,User,Therapist] triangle2[Client,User,Therapist,r] 1 triangle2[CertificateTemplate,User,Therapist,r] 1
 zV[r,lb,User,Therapist] triangle2[Course,User,Therapist,r] 1 triangle2[Role,User,Therapist,r] 1
 zV[r,lb,User,Therapist] triangle2[User,Therapist,Instructor,r] 1
 zV[c,lb,User,Therapist] choose1[User,Therapist] 1 axisVdifLB[c,User,Therapist] -53
 zV[c,lb,User,Therapist] DistAxisLB1[r,User,Therapist] 8.5 triangle1[User,CourseSession,Therapist,c] 1
 zV[c,lb,User,Therapist] triangle1[User,Clinic,Therapist,c] 1 triangle1[User,Location,Therapist,c] 1
 zV[c,lb,User,Therapist] triangle2[Client,User,Therapist,c] 1 triangle2[CertificateTemplate,User,Therapist,c] 1
 zV[c,lb,User,Therapist] triangle2[Course,User,Therapist,c] 1 triangle2[Role,User,Therapist,c] 1
 zV[c,lb,User,Therapist] triangle2[User,Therapist,Instructor,c] 1
 zV[r,rt,User,Instructor] choose1[User,Instructor] 1 axisVdifRT[r,User,Instructor] -52
 zV[r,rt,User,Instructor] DistAxisLB1[c,User,Instructor] 11.5 triangle1[Client,User,Instructor,r] 1
 zV[r,rt,User,Instructor] triangle1[CertificateTemplate,User,Instructor,r] 1 triangle1[Course,User,Instructor,r] 1
 zV[r,rt,User,Instructor] triangle1[Role,User,Instructor,r] 1 triangle2[User,CourseSession,Instructor,r] 1
 zV[r,rt,User,Instructor] triangle2[User,Clinic,Instructor,r] 1 triangle2[User,Location,Instructor,r] 1
 zV[r,rt,User,Instructor] triangle2[User,Therapist,Instructor,r] 1
 zV[c,rt,User,Instructor] choose1[User,Instructor] 1 axisVdifRT[c,User,Instructor] -54
 zV[c,rt,User,Instructor] DistAxisLB1[r,User,Instructor] 8.5 triangle1[Client,User,Instructor,c] 1
 zV[c,rt,User,Instructor] triangle1[CertificateTemplate,User,Instructor,c] 1 triangle1[Course,User,Instructor,c] 1
 zV[c,rt,User,Instructor] triangle1[Role,User,Instructor,c] 1 triangle2[User,CourseSession,Instructor,c] 1
 zV[c,rt,User,Instructor] triangle2[User,Clinic,Instructor,c] 1 triangle2[User,Location,Instructor,c] 1
 zV[c,rt,User,Instructor] triangle2[User,Therapist,Instructor,c] 1
 zV[r,lb,User,Instructor] choose1[User,Instructor] 1 axisVdifLB[r,User,Instructor] -49
 zV[r,lb,User,Instructor] DistAxisLB1[c,User,Instructor] 11.5 triangle1[User,CourseSession,Instructor,r] 1
 zV[r,lb,User,Instructor] triangle1[User,Clinic,Instructor,r] 1 triangle1[User,Location,Instructor,r] 1
 zV[r,lb,User,Instructor] triangle1[User,Therapist,Instructor,r] 1 triangle2[Client,User,Instructor,r] 1
 zV[r,lb,User,Instructor] triangle2[CertificateTemplate,User,Instructor,r] 1 triangle2[Course,User,Instructor,r] 1
 zV[r,lb,User,Instructor] triangle2[Role,User,Instructor,r] 1
 zV[c,lb,User,Instructor] choose1[User,Instructor] 1 axisVdifLB[c,User,Instructor] -53
 zV[c,lb,User,Instructor] DistAxisLB1[r,User,Instructor] 8.5 triangle1[User,CourseSession,Instructor,c] 1
 zV[c,lb,User,Instructor] triangle1[User,Clinic,Instructor,c] 1 triangle1[User,Location,Instructor,c] 1
 zV[c,lb,User,Instructor] triangle1[User,Therapist,Instructor,c] 1 triangle2[Client,User,Instructor,c] 1
 zV[c,lb,User,Instructor] triangle2[CertificateTemplate,User,Instructor,c] 1 triangle2[Course,User,Instructor,c] 1
 zV[c,lb,User,Instructor] triangle2[Role,User,Instructor,c] 1
 zV[r,rt,CourseSession,Clinic] choose1[CourseSession,Clinic] 1 axisVdifRT[r,CourseSession,Clinic] -50
 zV[r,rt,CourseSession,Clinic] triangle1[Client,CourseSession,Clinic,r] 1 triangle1[CertificateTemplate,CourseSession,Clinic,r] 1
 zV[r,rt,CourseSession,Clinic] triangle1[Course,CourseSession,Clinic,r] 1 triangle1[Role,CourseSession,Clinic,r] 1
 zV[r,rt,CourseSession,Clinic] triangle1[User,CourseSession,Clinic,r] 1 triangle1[CourseSession,Clinic,Location,r] 1
 zV[r,rt,CourseSession,Clinic] triangle1[CourseSession,Clinic,Therapist,r] 1 triangle1[CourseSession,Clinic,Instructor,r] 1
 zV[c,rt,CourseSession,Clinic] choose1[CourseSession,Clinic] 1 axisVdifRT[c,CourseSession,Clinic] -53
 zV[c,rt,CourseSession,Clinic] triangle1[Client,CourseSession,Clinic,c] 1 triangle1[CertificateTemplate,CourseSession,Clinic,c] 1
 zV[c,rt,CourseSession,Clinic] triangle1[Course,CourseSession,Clinic,c] 1 triangle1[Role,CourseSession,Clinic,c] 1
 zV[c,rt,CourseSession,Clinic] triangle1[User,CourseSession,Clinic,c] 1 triangle1[CourseSession,Clinic,Location,c] 1
 zV[c,rt,CourseSession,Clinic] triangle1[CourseSession,Clinic,Therapist,c] 1 triangle1[CourseSession,Clinic,Instructor,c] 1
 zV[r,lb,CourseSession,Clinic] choose1[CourseSession,Clinic] 1 axisVdifLB[r,CourseSession,Clinic] -51
 zV[r,lb,CourseSession,Clinic] triangle2[Client,CourseSession,Clinic,r] 1 triangle2[CertificateTemplate,CourseSession,Clinic,r] 1
 zV[r,lb,CourseSession,Clinic] triangle2[Course,CourseSession,Clinic,r] 1 triangle2[Role,CourseSession,Clinic,r] 1
 zV[r,lb,CourseSession,Clinic] triangle2[User,CourseSession,Clinic,r] 1 triangle2[CourseSession,Clinic,Location,r] 1
 zV[r,lb,CourseSession,Clinic] triangle2[CourseSession,Clinic,Therapist,r] 1 triangle2[CourseSession,Clinic,Instructor,r] 1
 zV[c,lb,CourseSession,Clinic] choose1[CourseSession,Clinic] 1 axisVdifLB[c,CourseSession,Clinic] -56
 zV[c,lb,CourseSession,Clinic] triangle2[Client,CourseSession,Clinic,c] 1 triangle2[CertificateTemplate,CourseSession,Clinic,c] 1
 zV[c,lb,CourseSession,Clinic] triangle2[Course,CourseSession,Clinic,c] 1 triangle2[Role,CourseSession,Clinic,c] 1
 zV[c,lb,CourseSession,Clinic] triangle2[User,CourseSession,Clinic,c] 1 triangle2[CourseSession,Clinic,Location,c] 1
 zV[c,lb,CourseSession,Clinic] triangle2[CourseSession,Clinic,Therapist,c] 1 triangle2[CourseSession,Clinic,Instructor,c] 1
 zV[r,rt,CourseSession,Location] choose1[CourseSession,Location] 1 axisVdifRT[r,CourseSession,Location] -51
 zV[r,rt,CourseSession,Location] DistAxisLB1[c,Location,CourseSession] 12.5 triangle1[Client,CourseSession,Location,r] 1
 zV[r,rt,CourseSession,Location] triangle1[CertificateTemplate,CourseSession,Location,r] 1 triangle1[Course,CourseSession,Location,r] 1
 zV[r,rt,CourseSession,Location] triangle1[Role,CourseSession,Location,r] 1 triangle1[User,CourseSession,Location,r] 1
 zV[r,rt,CourseSession,Location] triangle1[CourseSession,Location,Therapist,r] 1 triangle1[CourseSession,Location,Instructor,r] 1
 zV[r,rt,CourseSession,Location] triangle2[CourseSession,Clinic,Location,r] 1
 zV[c,rt,CourseSession,Location] choose1[CourseSession,Location] 1 axisVdifRT[c,CourseSession,Location] -53
 zV[c,rt,CourseSession,Location] DistAxisLB1[r,Location,CourseSession] 9 triangle1[Client,CourseSession,Location,c] 1
 zV[c,rt,CourseSession,Location] triangle1[CertificateTemplate,CourseSession,Location,c] 1 triangle1[Course,CourseSession,Location,c] 1
 zV[c,rt,CourseSession,Location] triangle1[Role,CourseSession,Location,c] 1 triangle1[User,CourseSession,Location,c] 1
 zV[c,rt,CourseSession,Location] triangle1[CourseSession,Location,Therapist,c] 1 triangle1[CourseSession,Location,Instructor,c] 1
 zV[c,rt,CourseSession,Location] triangle2[CourseSession,Clinic,Location,c] 1
 zV[r,lb,CourseSession,Location] choose1[CourseSession,Location] 1 axisVdifLB[r,CourseSession,Location] -51
 zV[r,lb,CourseSession,Location] DistAxisLB1[c,Location,CourseSession] 12.5 triangle1[CourseSession,Clinic,Location,r] 1
 zV[r,lb,CourseSession,Location] triangle2[Client,CourseSession,Location,r] 1 triangle2[CertificateTemplate,CourseSession,Location,r] 1
 zV[r,lb,CourseSession,Location] triangle2[Course,CourseSession,Location,r] 1 triangle2[Role,CourseSession,Location,r] 1
 zV[r,lb,CourseSession,Location] triangle2[User,CourseSession,Location,r] 1 triangle2[CourseSession,Location,Therapist,r] 1
 zV[r,lb,CourseSession,Location] triangle2[CourseSession,Location,Instructor,r] 1
 zV[c,lb,CourseSession,Location] choose1[CourseSession,Location] 1 axisVdifLB[c,CourseSession,Location] -56
 zV[c,lb,CourseSession,Location] DistAxisLB1[r,Location,CourseSession] 9 triangle1[CourseSession,Clinic,Location,c] 1
 zV[c,lb,CourseSession,Location] triangle2[Client,CourseSession,Location,c] 1 triangle2[CertificateTemplate,CourseSession,Location,c] 1
 zV[c,lb,CourseSession,Location] triangle2[Course,CourseSession,Location,c] 1 triangle2[Role,CourseSession,Location,c] 1
 zV[c,lb,CourseSession,Location] triangle2[User,CourseSession,Location,c] 1 triangle2[CourseSession,Location,Therapist,c] 1
 zV[c,lb,CourseSession,Location] triangle2[CourseSession,Location,Instructor,c] 1
 zV[r,rt,CourseSession,Therapist] choose1[CourseSession,Therapist] 1 axisVdifRT[r,CourseSession,Therapist] -52
 zV[r,rt,CourseSession,Therapist] DistAxisLB1[c,Therapist,CourseSession] 12.5 triangle1[Client,CourseSession,Therapist,r] 1
 zV[r,rt,CourseSession,Therapist] triangle1[CertificateTemplate,CourseSession,Therapist,r] 1 triangle1[Course,CourseSession,Therapist,r] 1
 zV[r,rt,CourseSession,Therapist] triangle1[Role,CourseSession,Therapist,r] 1 triangle1[User,CourseSession,Therapist,r] 1
 zV[r,rt,CourseSession,Therapist] triangle1[CourseSession,Therapist,Instructor,r] 1 triangle2[CourseSession,Clinic,Therapist,r] 1
 zV[r,rt,CourseSession,Therapist] triangle2[CourseSession,Location,Therapist,r] 1
 zV[c,rt,CourseSession,Therapist] choose1[CourseSession,Therapist] 1 axisVdifRT[c,CourseSession,Therapist] -53
 zV[c,rt,CourseSession,Therapist] DistAxisLB1[r,Therapist,CourseSession] 9.5 triangle1[Client,CourseSession,Therapist,c] 1
 zV[c,rt,CourseSession,Therapist] triangle1[CertificateTemplate,CourseSession,Therapist,c] 1 triangle1[Course,CourseSession,Therapist,c] 1
 zV[c,rt,CourseSession,Therapist] triangle1[Role,CourseSession,Therapist,c] 1 triangle1[User,CourseSession,Therapist,c] 1
 zV[c,rt,CourseSession,Therapist] triangle1[CourseSession,Therapist,Instructor,c] 1 triangle2[CourseSession,Clinic,Therapist,c] 1
 zV[c,rt,CourseSession,Therapist] triangle2[CourseSession,Location,Therapist,c] 1
 zV[r,lb,CourseSession,Therapist] choose1[CourseSession,Therapist] 1 axisVdifLB[r,CourseSession,Therapist] -51
 zV[r,lb,CourseSession,Therapist] DistAxisLB1[c,Therapist,CourseSession] 12.5 triangle1[CourseSession,Clinic,Therapist,r] 1
 zV[r,lb,CourseSession,Therapist] triangle1[CourseSession,Location,Therapist,r] 1 triangle2[Client,CourseSession,Therapist,r] 1
 zV[r,lb,CourseSession,Therapist] triangle2[CertificateTemplate,CourseSession,Therapist,r] 1 triangle2[Course,CourseSession,Therapist,r] 1
 zV[r,lb,CourseSession,Therapist] triangle2[Role,CourseSession,Therapist,r] 1 triangle2[User,CourseSession,Therapist,r] 1
 zV[r,lb,CourseSession,Therapist] triangle2[CourseSession,Therapist,Instructor,r] 1
 zV[c,lb,CourseSession,Therapist] choose1[CourseSession,Therapist] 1 axisVdifLB[c,CourseSession,Therapist] -56
 zV[c,lb,CourseSession,Therapist] DistAxisLB1[r,Therapist,CourseSession] 9.5 triangle1[CourseSession,Clinic,Therapist,c] 1
 zV[c,lb,CourseSession,Therapist] triangle1[CourseSession,Location,Therapist,c] 1 triangle2[Client,CourseSession,Therapist,c] 1
 zV[c,lb,CourseSession,Therapist] triangle2[CertificateTemplate,CourseSession,Therapist,c] 1 triangle2[Course,CourseSession,Therapist,c] 1
 zV[c,lb,CourseSession,Therapist] triangle2[Role,CourseSession,Therapist,c] 1 triangle2[User,CourseSession,Therapist,c] 1
 zV[c,lb,CourseSession,Therapist] triangle2[CourseSession,Therapist,Instructor,c] 1
 zV[r,rt,CourseSession,Instructor] choose1[CourseSession,Instructor] 1 axisVdifRT[r,CourseSession,Instructor] -52
 zV[r,rt,CourseSession,Instructor] DistAxisLB1[c,CourseSession,Instructor] 13 triangle1[Client,CourseSession,Instructor,r] 1
 zV[r,rt,CourseSession,Instructor] triangle1[CertificateTemplate,CourseSession,Instructor,r] 1 triangle1[Course,CourseSession,Instructor,r] 1
 zV[r,rt,CourseSession,Instructor] triangle1[Role,CourseSession,Instructor,r] 1 triangle1[User,CourseSession,Instructor,r] 1
 zV[r,rt,CourseSession,Instructor] triangle2[CourseSession,Clinic,Instructor,r] 1 triangle2[CourseSession,Location,Instructor,r] 1
 zV[r,rt,CourseSession,Instructor] triangle2[CourseSession,Therapist,Instructor,r] 1
 zV[c,rt,CourseSession,Instructor] choose1[CourseSession,Instructor] 1 axisVdifRT[c,CourseSession,Instructor] -54
 zV[c,rt,CourseSession,Instructor] DistAxisLB1[r,CourseSession,Instructor] 9.5 triangle1[Client,CourseSession,Instructor,c] 1
 zV[c,rt,CourseSession,Instructor] triangle1[CertificateTemplate,CourseSession,Instructor,c] 1 triangle1[Course,CourseSession,Instructor,c] 1
 zV[c,rt,CourseSession,Instructor] triangle1[Role,CourseSession,Instructor,c] 1 triangle1[User,CourseSession,Instructor,c] 1
 zV[c,rt,CourseSession,Instructor] triangle2[CourseSession,Clinic,Instructor,c] 1 triangle2[CourseSession,Location,Instructor,c] 1
 zV[c,rt,CourseSession,Instructor] triangle2[CourseSession,Therapist,Instructor,c] 1
 zV[r,lb,CourseSession,Instructor] choose1[CourseSession,Instructor] 1 axisVdifLB[r,CourseSession,Instructor] -51
 zV[r,lb,CourseSession,Instructor] DistAxisLB1[c,CourseSession,Instructor] 13 triangle1[CourseSession,Clinic,Instructor,r] 1
 zV[r,lb,CourseSession,Instructor] triangle1[CourseSession,Location,Instructor,r] 1 triangle1[CourseSession,Therapist,Instructor,r] 1
 zV[r,lb,CourseSession,Instructor] triangle2[Client,CourseSession,Instructor,r] 1 triangle2[CertificateTemplate,CourseSession,Instructor,r] 1
 zV[r,lb,CourseSession,Instructor] triangle2[Course,CourseSession,Instructor,r] 1 triangle2[Role,CourseSession,Instructor,r] 1
 zV[r,lb,CourseSession,Instructor] triangle2[User,CourseSession,Instructor,r] 1
 zV[c,lb,CourseSession,Instructor] choose1[CourseSession,Instructor] 1 axisVdifLB[c,CourseSession,Instructor] -56
 zV[c,lb,CourseSession,Instructor] DistAxisLB1[r,CourseSession,Instructor] 9.5 triangle1[CourseSession,Clinic,Instructor,c] 1
 zV[c,lb,CourseSession,Instructor] triangle1[CourseSession,Location,Instructor,c] 1 triangle1[CourseSession,Therapist,Instructor,c] 1
 zV[c,lb,CourseSession,Instructor] triangle2[Client,CourseSession,Instructor,c] 1 triangle2[CertificateTemplate,CourseSession,Instructor,c] 1
 zV[c,lb,CourseSession,Instructor] triangle2[Course,CourseSession,Instructor,c] 1 triangle2[Role,CourseSession,Instructor,c] 1
 zV[c,lb,CourseSession,Instructor] triangle2[User,CourseSession,Instructor,c] 1
 zV[r,rt,Clinic,Location] choose1[Clinic,Location] 1 axisVdifRT[r,Clinic,Location] -51
 zV[r,rt,Clinic,Location] DistAxisLB1[c,Clinic,Location] 11 triangle1[Client,Clinic,Location,r] 1
 zV[r,rt,Clinic,Location] triangle1[CertificateTemplate,Clinic,Location,r] 1 triangle1[Course,Clinic,Location,r] 1
 zV[r,rt,Clinic,Location] triangle1[Role,Clinic,Location,r] 1 triangle1[User,Clinic,Location,r] 1
 zV[r,rt,Clinic,Location] triangle1[CourseSession,Clinic,Location,r] 1 triangle1[Clinic,Location,Therapist,r] 1
 zV[r,rt,Clinic,Location] triangle1[Clinic,Location,Instructor,r] 1
 zV[c,rt,Clinic,Location] choose1[Clinic,Location] 1 axisVdifRT[c,Clinic,Location] -53
 zV[c,rt,Clinic,Location] DistAxisLB1[r,Clinic,Location] 8.5 triangle1[Client,Clinic,Location,c] 1
 zV[c,rt,Clinic,Location] triangle1[CertificateTemplate,Clinic,Location,c] 1 triangle1[Course,Clinic,Location,c] 1
 zV[c,rt,Clinic,Location] triangle1[Role,Clinic,Location,c] 1 triangle1[User,Clinic,Location,c] 1
 zV[c,rt,Clinic,Location] triangle1[CourseSession,Clinic,Location,c] 1 triangle1[Clinic,Location,Therapist,c] 1
 zV[c,rt,Clinic,Location] triangle1[Clinic,Location,Instructor,c] 1
 zV[r,lb,Clinic,Location] choose1[Clinic,Location] 1 axisVdifLB[r,Clinic,Location] -50
 zV[r,lb,Clinic,Location] DistAxisLB1[c,Clinic,Location] 11 triangle2[Client,Clinic,Location,r] 1
 zV[r,lb,Clinic,Location] triangle2[CertificateTemplate,Clinic,Location,r] 1 triangle2[Course,Clinic,Location,r] 1
 zV[r,lb,Clinic,Location] triangle2[Role,Clinic,Location,r] 1 triangle2[User,Clinic,Location,r] 1
 zV[r,lb,Clinic,Location] triangle2[CourseSession,Clinic,Location,r] 1 triangle2[Clinic,Location,Therapist,r] 1
 zV[r,lb,Clinic,Location] triangle2[Clinic,Location,Instructor,r] 1
 zV[c,lb,Clinic,Location] choose1[Clinic,Location] 1 axisVdifLB[c,Clinic,Location] -53
 zV[c,lb,Clinic,Location] DistAxisLB1[r,Clinic,Location] 8.5 triangle2[Client,Clinic,Location,c] 1
 zV[c,lb,Clinic,Location] triangle2[CertificateTemplate,Clinic,Location,c] 1 triangle2[Course,Clinic,Location,c] 1
 zV[c,lb,Clinic,Location] triangle2[Role,Clinic,Location,c] 1 triangle2[User,Clinic,Location,c] 1
 zV[c,lb,Clinic,Location] triangle2[CourseSession,Clinic,Location,c] 1 triangle2[Clinic,Location,Therapist,c] 1
 zV[c,lb,Clinic,Location] triangle2[Clinic,Location,Instructor,c] 1
 zV[r,rt,Clinic,Therapist] choose1[Clinic,Therapist] 1 axisVdifRT[r,Clinic,Therapist] -52
 zV[r,rt,Clinic,Therapist] DistAxisLB1[c,Clinic,Therapist] 11 triangle1[Client,Clinic,Therapist,r] 1
 zV[r,rt,Clinic,Therapist] triangle1[CertificateTemplate,Clinic,Therapist,r] 1 triangle1[Course,Clinic,Therapist,r] 1
 zV[r,rt,Clinic,Therapist] triangle1[Role,Clinic,Therapist,r] 1 triangle1[User,Clinic,Therapist,r] 1
 zV[r,rt,Clinic,Therapist] triangle1[CourseSession,Clinic,Therapist,r] 1 triangle1[Clinic,Therapist,Instructor,r] 1
 zV[r,rt,Clinic,Therapist] triangle2[Clinic,Location,Therapist,r] 1
 zV[c,rt,Clinic,Therapist] choose1[Clinic,Therapist] 1 axisVdifRT[c,Clinic,Therapist] -53
 zV[c,rt,Clinic,Therapist] DistAxisLB1[r,Clinic,Therapist] 9 triangle1[Client,Clinic,Therapist,c] 1
 zV[c,rt,Clinic,Therapist] triangle1[CertificateTemplate,Clinic,Therapist,c] 1 triangle1[Course,Clinic,Therapist,c] 1
 zV[c,rt,Clinic,Therapist] triangle1[Role,Clinic,Therapist,c] 1 triangle1[User,Clinic,Therapist,c] 1
 zV[c,rt,Clinic,Therapist] triangle1[CourseSession,Clinic,Therapist,c] 1 triangle1[Clinic,Therapist,Instructor,c] 1
 zV[c,rt,Clinic,Therapist] triangle2[Clinic,Location,Therapist,c] 1
 zV[r,lb,Clinic,Therapist] choose1[Clinic,Therapist] 1 axisVdifLB[r,Clinic,Therapist] -50
 zV[r,lb,Clinic,Therapist] DistAxisLB1[c,Clinic,Therapist] 11 triangle1[Clinic,Location,Therapist,r] 1
 zV[r,lb,Clinic,Therapist] triangle2[Client,Clinic,Therapist,r] 1 triangle2[CertificateTemplate,Clinic,Therapist,r] 1
 zV[r,lb,Clinic,Therapist] triangle2[Course,Clinic,Therapist,r] 1 triangle2[Role,Clinic,Therapist,r] 1
 zV[r,lb,Clinic,Therapist] triangle2[User,Clinic,Therapist,r] 1 triangle2[CourseSession,Clinic,Therapist,r] 1
 zV[r,lb,Clinic,Therapist] triangle2[Clinic,Therapist,Instructor,r] 1
 zV[c,lb,Clinic,Therapist] choose1[Clinic,Therapist] 1 axisVdifLB[c,Clinic,Therapist] -53
 zV[c,lb,Clinic,Therapist] DistAxisLB1[r,Clinic,Therapist] 9 triangle1[Clinic,Location,Therapist,c] 1
 zV[c,lb,Clinic,Therapist] triangle2[Client,Clinic,Therapist,c] 1 triangle2[CertificateTemplate,Clinic,Therapist,c] 1
 zV[c,lb,Clinic,Therapist] triangle2[Course,Clinic,Therapist,c] 1 triangle2[Role,Clinic,Therapist,c] 1
 zV[c,lb,Clinic,Therapist] triangle2[User,Clinic,Therapist,c] 1 triangle2[CourseSession,Clinic,Therapist,c] 1
 zV[c,lb,Clinic,Therapist] triangle2[Clinic,Therapist,Instructor,c] 1
 zV[r,rt,Clinic,Instructor] choose1[Clinic,Instructor] 1 axisVdifRT[r,Clinic,Instructor] -52
 zV[r,rt,Clinic,Instructor] triangle1[Client,Clinic,Instructor,r] 1 triangle1[CertificateTemplate,Clinic,Instructor,r] 1
 zV[r,rt,Clinic,Instructor] triangle1[Course,Clinic,Instructor,r] 1 triangle1[Role,Clinic,Instructor,r] 1
 zV[r,rt,Clinic,Instructor] triangle1[User,Clinic,Instructor,r] 1 triangle1[CourseSession,Clinic,Instructor,r] 1
 zV[r,rt,Clinic,Instructor] triangle2[Clinic,Location,Instructor,r] 1 triangle2[Clinic,Therapist,Instructor,r] 1
 zV[c,rt,Clinic,Instructor] choose1[Clinic,Instructor] 1 axisVdifRT[c,Clinic,Instructor] -54
 zV[c,rt,Clinic,Instructor] triangle1[Client,Clinic,Instructor,c] 1 triangle1[CertificateTemplate,Clinic,Instructor,c] 1
 zV[c,rt,Clinic,Instructor] triangle1[Course,Clinic,Instructor,c] 1 triangle1[Role,Clinic,Instructor,c] 1
 zV[c,rt,Clinic,Instructor] triangle1[User,Clinic,Instructor,c] 1 triangle1[CourseSession,Clinic,Instructor,c] 1
 zV[c,rt,Clinic,Instructor] triangle2[Clinic,Location,Instructor,c] 1 triangle2[Clinic,Therapist,Instructor,c] 1
 zV[r,lb,Clinic,Instructor] choose1[Clinic,Instructor] 1 axisVdifLB[r,Clinic,Instructor] -50
 zV[r,lb,Clinic,Instructor] triangle1[Clinic,Location,Instructor,r] 1 triangle1[Clinic,Therapist,Instructor,r] 1
 zV[r,lb,Clinic,Instructor] triangle2[Client,Clinic,Instructor,r] 1 triangle2[CertificateTemplate,Clinic,Instructor,r] 1
 zV[r,lb,Clinic,Instructor] triangle2[Course,Clinic,Instructor,r] 1 triangle2[Role,Clinic,Instructor,r] 1
 zV[r,lb,Clinic,Instructor] triangle2[User,Clinic,Instructor,r] 1 triangle2[CourseSession,Clinic,Instructor,r] 1
 zV[c,lb,Clinic,Instructor] choose1[Clinic,Instructor] 1 axisVdifLB[c,Clinic,Instructor] -53
 zV[c,lb,Clinic,Instructor] triangle1[Clinic,Location,Instructor,c] 1 triangle1[Clinic,Therapist,Instructor,c] 1
 zV[c,lb,Clinic,Instructor] triangle2[Client,Clinic,Instructor,c] 1 triangle2[CertificateTemplate,Clinic,Instructor,c] 1
 zV[c,lb,Clinic,Instructor] triangle2[Course,Clinic,Instructor,c] 1 triangle2[Role,Clinic,Instructor,c] 1
 zV[c,lb,Clinic,Instructor] triangle2[User,Clinic,Instructor,c] 1 triangle2[CourseSession,Clinic,Instructor,c] 1
 zV[r,rt,Location,Therapist] choose1[Location,Therapist] 1 axisVdifRT[r,Location,Therapist] -52
 zV[r,rt,Location,Therapist] DistAxisLB1[c,Location,Therapist] 11 triangle1[Client,Location,Therapist,r] 1
 zV[r,rt,Location,Therapist] triangle1[CertificateTemplate,Location,Therapist,r] 1 triangle1[Course,Location,Therapist,r] 1
 zV[r,rt,Location,Therapist] triangle1[Role,Location,Therapist,r] 1 triangle1[User,Location,Therapist,r] 1
 zV[r,rt,Location,Therapist] triangle1[CourseSession,Location,Therapist,r] 1 triangle1[Clinic,Location,Therapist,r] 1
 zV[r,rt,Location,Therapist] triangle1[Location,Therapist,Instructor,r] 1
 zV[c,rt,Location,Therapist] choose1[Location,Therapist] 1 axisVdifRT[c,Location,Therapist] -53
 zV[c,rt,Location,Therapist] DistAxisLB1[r,Location,Therapist] 9.5 triangle1[Client,Location,Therapist,c] 1
 zV[c,rt,Location,Therapist] triangle1[CertificateTemplate,Location,Therapist,c] 1 triangle1[Course,Location,Therapist,c] 1
 zV[c,rt,Location,Therapist] triangle1[Role,Location,Therapist,c] 1 triangle1[User,Location,Therapist,c] 1
 zV[c,rt,Location,Therapist] triangle1[CourseSession,Location,Therapist,c] 1 triangle1[Clinic,Location,Therapist,c] 1
 zV[c,rt,Location,Therapist] triangle1[Location,Therapist,Instructor,c] 1
 zV[r,lb,Location,Therapist] choose1[Location,Therapist] 1 axisVdifLB[r,Location,Therapist] -51
 zV[r,lb,Location,Therapist] DistAxisLB1[c,Location,Therapist] 11 triangle2[Client,Location,Therapist,r] 1
 zV[r,lb,Location,Therapist] triangle2[CertificateTemplate,Location,Therapist,r] 1 triangle2[Course,Location,Therapist,r] 1
 zV[r,lb,Location,Therapist] triangle2[Role,Location,Therapist,r] 1 triangle2[User,Location,Therapist,r] 1
 zV[r,lb,Location,Therapist] triangle2[CourseSession,Location,Therapist,r] 1 triangle2[Clinic,Location,Therapist,r] 1
 zV[r,lb,Location,Therapist] triangle2[Location,Therapist,Instructor,r] 1
 zV[c,lb,Location,Therapist] choose1[Location,Therapist] 1 axisVdifLB[c,Location,Therapist] -53
 zV[c,lb,Location,Therapist] DistAxisLB1[r,Location,Therapist] 9.5 triangle2[Client,Location,Therapist,c] 1
 zV[c,lb,Location,Therapist] triangle2[CertificateTemplate,Location,Therapist,c] 1 triangle2[Course,Location,Therapist,c] 1
 zV[c,lb,Location,Therapist] triangle2[Role,Location,Therapist,c] 1 triangle2[User,Location,Therapist,c] 1
 zV[c,lb,Location,Therapist] triangle2[CourseSession,Location,Therapist,c] 1 triangle2[Clinic,Location,Therapist,c] 1
 zV[c,lb,Location,Therapist] triangle2[Location,Therapist,Instructor,c] 1
 zV[r,rt,Location,Instructor] choose1[Location,Instructor] 1 axisVdifRT[r,Location,Instructor] -52
 zV[r,rt,Location,Instructor] DistAxisLB1[c,Instructor,Location] 11.5 triangle1[Client,Location,Instructor,r] 1
 zV[r,rt,Location,Instructor] triangle1[CertificateTemplate,Location,Instructor,r] 1 triangle1[Course,Location,Instructor,r] 1
 zV[r,rt,Location,Instructor] triangle1[Role,Location,Instructor,r] 1 triangle1[User,Location,Instructor,r] 1
 zV[r,rt,Location,Instructor] triangle1[CourseSession,Location,Instructor,r] 1 triangle1[Clinic,Location,Instructor,r] 1
 zV[r,rt,Location,Instructor] triangle2[Location,Therapist,Instructor,r] 1
 zV[c,rt,Location,Instructor] choose1[Location,Instructor] 1 axisVdifRT[c,Location,Instructor] -54
 zV[c,rt,Location,Instructor] DistAxisLB1[r,Instructor,Location] 9.5 triangle1[Client,Location,Instructor,c] 1
 zV[c,rt,Location,Instructor] triangle1[CertificateTemplate,Location,Instructor,c] 1 triangle1[Course,Location,Instructor,c] 1
 zV[c,rt,Location,Instructor] triangle1[Role,Location,Instructor,c] 1 triangle1[User,Location,Instructor,c] 1
 zV[c,rt,Location,Instructor] triangle1[CourseSession,Location,Instructor,c] 1 triangle1[Clinic,Location,Instructor,c] 1
 zV[c,rt,Location,Instructor] triangle2[Location,Therapist,Instructor,c] 1
 zV[r,lb,Location,Instructor] choose1[Location,Instructor] 1 axisVdifLB[r,Location,Instructor] -51
 zV[r,lb,Location,Instructor] DistAxisLB1[c,Instructor,Location] 11.5 triangle1[Location,Therapist,Instructor,r] 1
 zV[r,lb,Location,Instructor] triangle2[Client,Location,Instructor,r] 1 triangle2[CertificateTemplate,Location,Instructor,r] 1
 zV[r,lb,Location,Instructor] triangle2[Course,Location,Instructor,r] 1 triangle2[Role,Location,Instructor,r] 1
 zV[r,lb,Location,Instructor] triangle2[User,Location,Instructor,r] 1 triangle2[CourseSession,Location,Instructor,r] 1
 zV[r,lb,Location,Instructor] triangle2[Clinic,Location,Instructor,r] 1
 zV[c,lb,Location,Instructor] choose1[Location,Instructor] 1 axisVdifLB[c,Location,Instructor] -53
 zV[c,lb,Location,Instructor] DistAxisLB1[r,Instructor,Location] 9.5 triangle1[Location,Therapist,Instructor,c] 1
 zV[c,lb,Location,Instructor] triangle2[Client,Location,Instructor,c] 1 triangle2[CertificateTemplate,Location,Instructor,c] 1
 zV[c,lb,Location,Instructor] triangle2[Course,Location,Instructor,c] 1 triangle2[Role,Location,Instructor,c] 1
 zV[c,lb,Location,Instructor] triangle2[User,Location,Instructor,c] 1 triangle2[CourseSession,Location,Instructor,c] 1
 zV[c,lb,Location,Instructor] triangle2[Clinic,Location,Instructor,c] 1
 zV[r,rt,Therapist,Instructor] choose1[Therapist,Instructor] 1 axisVdifRT[r,Therapist,Instructor] -52
 zV[r,rt,Therapist,Instructor] triangle1[Client,Therapist,Instructor,r] 1 triangle1[CertificateTemplate,Therapist,Instructor,r] 1
 zV[r,rt,Therapist,Instructor] triangle1[Course,Therapist,Instructor,r] 1 triangle1[Role,Therapist,Instructor,r] 1
 zV[r,rt,Therapist,Instructor] triangle1[User,Therapist,Instructor,r] 1 triangle1[CourseSession,Therapist,Instructor,r] 1
 zV[r,rt,Therapist,Instructor] triangle1[Clinic,Therapist,Instructor,r] 1 triangle1[Location,Therapist,Instructor,r] 1
 zV[c,rt,Therapist,Instructor] choose1[Therapist,Instructor] 1 axisVdifRT[c,Therapist,Instructor] -54
 zV[c,rt,Therapist,Instructor] triangle1[Client,Therapist,Instructor,c] 1 triangle1[CertificateTemplate,Therapist,Instructor,c] 1
 zV[c,rt,Therapist,Instructor] triangle1[Course,Therapist,Instructor,c] 1 triangle1[Role,Therapist,Instructor,c] 1
 zV[c,rt,Therapist,Instructor] triangle1[User,Therapist,Instructor,c] 1 triangle1[CourseSession,Therapist,Instructor,c] 1
 zV[c,rt,Therapist,Instructor] triangle1[Clinic,Therapist,Instructor,c] 1 triangle1[Location,Therapist,Instructor,c] 1
 zV[r,lb,Therapist,Instructor] choose1[Therapist,Instructor] 1 axisVdifLB[r,Therapist,Instructor] -52
 zV[r,lb,Therapist,Instructor] triangle2[Client,Therapist,Instructor,r] 1 triangle2[CertificateTemplate,Therapist,Instructor,r] 1
 zV[r,lb,Therapist,Instructor] triangle2[Course,Therapist,Instructor,r] 1 triangle2[Role,Therapist,Instructor,r] 1
 zV[r,lb,Therapist,Instructor] triangle2[User,Therapist,Instructor,r] 1 triangle2[CourseSession,Therapist,Instructor,r] 1
 zV[r,lb,Therapist,Instructor] triangle2[Clinic,Therapist,Instructor,r] 1 triangle2[Location,Therapist,Instructor,r] 1
 zV[c,lb,Therapist,Instructor] choose1[Therapist,Instructor] 1 axisVdifLB[c,Therapist,Instructor] -53
 zV[c,lb,Therapist,Instructor] triangle2[Client,Therapist,Instructor,c] 1 triangle2[CertificateTemplate,Therapist,Instructor,c] 1
 zV[c,lb,Therapist,Instructor] triangle2[Course,Therapist,Instructor,c] 1 triangle2[Role,Therapist,Instructor,c] 1
 zV[c,lb,Therapist,Instructor] triangle2[User,Therapist,Instructor,c] 1 triangle2[CourseSession,Therapist,Instructor,c] 1
 zV[c,lb,Therapist,Instructor] triangle2[Clinic,Therapist,Instructor,c] 1 triangle2[Location,Therapist,Instructor,c] 1
 M0000004 'MARKER' 'INTEND'
RHS
 RHS1 choose1[Client,CertificateTemplate] 1 choose1[Client,Course] 1
 RHS1 choose1[Client,Role] 1 choose1[Client,User] 1
 RHS1 choose1[Client,CourseSession] 1 choose1[Client,Clinic] 1
 RHS1 choose1[Client,Location] 1 choose1[Client,Therapist] 1
 RHS1 choose1[Client,Instructor] 1 choose1[CertificateTemplate,Course] 1
 RHS1 choose1[CertificateTemplate,Role] 1 choose1[CertificateTemplate,User] 1
 RHS1 choose1[CertificateTemplate,CourseSession] 1 choose1[CertificateTemplate,Clinic] 1
 RHS1 choose1[CertificateTemplate,Location] 1 choose1[CertificateTemplate,Therapist] 1
 RHS1 choose1[CertificateTemplate,Instructor] 1 choose1[Course,Role] 1
 RHS1 choose1[Course,User] 1 choose1[Course,CourseSession] 1
 RHS1 choose1[Course,Clinic] 1 choose1[Course,Location] 1
 RHS1 choose1[Course,Therapist] 1 choose1[Course,Instructor] 1
 RHS1 choose1[Role,User] 1 choose1[Role,CourseSession] 1
 RHS1 choose1[Role,Clinic] 1 choose1[Role,Location] 1
 RHS1 choose1[Role,Therapist] 1 choose1[Role,Instructor] 1
 RHS1 choose1[User,CourseSession] 1 choose1[User,Clinic] 1
 RHS1 choose1[User,Location] 1 choose1[User,Therapist] 1
 RHS1 choose1[User,Instructor] 1 choose1[CourseSession,Clinic] 1
 RHS1 choose1[CourseSession,Location] 1 choose1[CourseSession,Therapist] 1
 RHS1 choose1[CourseSession,Instructor] 1 choose1[Clinic,Location] 1
 RHS1 choose1[Clinic,Therapist] 1 choose1[Clinic,Instructor] 1
 RHS1 choose1[Location,Therapist] 1 choose1[Location,Instructor] 1
 RHS1 choose1[Therapist,Instructor] 1 centerDistAxis1[r,Client] 17.5
 RHS1 centerDistAxis1[r,CertificateTemplate] 18.5 centerDistAxis1[r,Course] 19
 RHS1 centerDistAxis1[r,Role] 19 centerDistAxis1[r,User] 18.5
 RHS1 centerDistAxis1[r,CourseSession] 17.5 centerDistAxis1[r,Clinic] 18
 RHS1 centerDistAxis1[r,Location] 17.5 centerDistAxis1[r,Therapist] 17
 RHS1 centerDistAxis1[r,Instructor] 17 centerDistAxis1[c,Client] 16.5
 RHS1 centerDistAxis1[c,CertificateTemplate] 16 centerDistAxis1[c,Course] 17
 RHS1 centerDistAxis1[c,Role] 17 centerDistAxis1[c,User] 16.5
 RHS1 centerDistAxis1[c,CourseSession] 15 centerDistAxis1[c,Clinic] 16.5
 RHS1 centerDistAxis1[c,Location] 16.5 centerDistAxis1[c,Therapist] 16.5
 RHS1 centerDistAxis1[c,Instructor] 16 centerDistAxis2[r,Client] -17.5
 RHS1 centerDistAxis2[r,CertificateTemplate] -18.5 centerDistAxis2[r,Course] -19
 RHS1 centerDistAxis2[r,Role] -19 centerDistAxis2[r,User] -18.5
 RHS1 centerDistAxis2[r,CourseSession] -17.5 centerDistAxis2[r,Clinic] -18
 RHS1 centerDistAxis2[r,Location] -17.5 centerDistAxis2[r,Therapist] -17
 RHS1 centerDistAxis2[r,Instructor] -17 centerDistAxis2[c,Client] -16.5
 RHS1 centerDistAxis2[c,CertificateTemplate] -16 centerDistAxis2[c,Course] -17
 RHS1 centerDistAxis2[c,Role] -17 centerDistAxis2[c,User] -16.5
 RHS1 centerDistAxis2[c,CourseSession] -15 centerDistAxis2[c,Clinic] -16.5
 RHS1 centerDistAxis2[c,Location] -16.5 centerDistAxis2[c,Therapist] -16.5
 RHS1 centerDistAxis2[c,Instructor] -16 axisVdifLB[r,Client,CertificateTemplate] -42
 RHS1 axisVdifLB[r,Client,Course] -42 axisVdifLB[r,Client,Role] -42
 RHS1 axisVdifLB[r,Client,User] -42 axisVdifLB[r,Client,CourseSession] -42
 RHS1 axisVdifLB[r,Client,Clinic] -42 axisVdifLB[r,Client,Location] -42
 RHS1 axisVdifLB[r,Client,Therapist] -42 axisVdifLB[r,Client,Instructor] -42
 RHS1 axisVdifLB[r,CertificateTemplate,Course] -42 axisVdifLB[r,CertificateTemplate,Role] -42
 RHS1 axisVdifLB[r,CertificateTemplate,User] -42 axisVdifLB[r,CertificateTemplate,CourseSession] -42
 RHS1 axisVdifLB[r,CertificateTemplate,Clinic] -42 axisVdifLB[r,CertificateTemplate,Location] -42
 RHS1 axisVdifLB[r,CertificateTemplate,Therapist] -42 axisVdifLB[r,CertificateTemplate,Instructor] -42
 RHS1 axisVdifLB[r,Course,Role] -42 axisVdifLB[r,Course,User] -42
 RHS1 axisVdifLB[r,Course,CourseSession] -42 axisVdifLB[r,Course,Clinic] -42
 RHS1 axisVdifLB[r,Course,Location] -42 axisVdifLB[r,Course,Therapist] -42
 RHS1 axisVdifLB[r,Course,Instructor] -42 axisVdifLB[r,Role,User] -42
 RHS1 axisVdifLB[r,Role,CourseSession] -42 axisVdifLB[r,Role,Clinic] -42
 RHS1 axisVdifLB[r,Role,Location] -42 axisVdifLB[r,Role,Therapist] -42
 RHS1 axisVdifLB[r,Role,Instructor] -42 axisVdifLB[r,User,CourseSession] -42
 RHS1 axisVdifLB[r,User,Clinic] -42 axisVdifLB[r,User,Location] -42
 RHS1 axisVdifLB[r,User,Therapist] -42 axisVdifLB[r,User,Instructor] -42
 RHS1 axisVdifLB[r,CourseSession,Clinic] -42 axisVdifLB[r,CourseSession,Location] -42
 RHS1 axisVdifLB[r,CourseSession,Therapist] -42 axisVdifLB[r,CourseSession,Instructor] -42
 RHS1 axisVdifLB[r,Clinic,Location] -42 axisVdifLB[r,Clinic,Therapist] -42
 RHS1 axisVdifLB[r,Clinic,Instructor] -42 axisVdifLB[r,Location,Therapist] -42
 RHS1 axisVdifLB[r,Location,Instructor] -42 axisVdifLB[r,Therapist,Instructor] -42
 RHS1 axisVdifLB[c,Client,CertificateTemplate] -42 axisVdifLB[c,Client,Course] -42
 RHS1 axisVdifLB[c,Client,Role] -42 axisVdifLB[c,Client,User] -42
 RHS1 axisVdifLB[c,Client,CourseSession] -42 axisVdifLB[c,Client,Clinic] -42
 RHS1 axisVdifLB[c,Client,Location] -42 axisVdifLB[c,Client,Therapist] -42
 RHS1 axisVdifLB[c,Client,Instructor] -42 axisVdifLB[c,CertificateTemplate,Course] -42
 RHS1 axisVdifLB[c,CertificateTemplate,Role] -42 axisVdifLB[c,CertificateTemplate,User] -42
 RHS1 axisVdifLB[c,CertificateTemplate,CourseSession] -42 axisVdifLB[c,CertificateTemplate,Clinic] -42
 RHS1 axisVdifLB[c,CertificateTemplate,Location] -42 axisVdifLB[c,CertificateTemplate,Therapist] -42
 RHS1 axisVdifLB[c,CertificateTemplate,Instructor] -42 axisVdifLB[c,Course,Role] -42
 RHS1 axisVdifLB[c,Course,User] -42 axisVdifLB[c,Course,CourseSession] -42
 RHS1 axisVdifLB[c,Course,Clinic] -42 axisVdifLB[c,Course,Location] -42
 RHS1 axisVdifLB[c,Course,Therapist] -42 axisVdifLB[c,Course,Instructor] -42
 RHS1 axisVdifLB[c,Role,User] -42 axisVdifLB[c,Role,CourseSession] -42
 RHS1 axisVdifLB[c,Role,Clinic] -42 axisVdifLB[c,Role,Location] -42
 RHS1 axisVdifLB[c,Role,Therapist] -42 axisVdifLB[c,Role,Instructor] -42
 RHS1 axisVdifLB[c,User,CourseSession] -42 axisVdifLB[c,User,Clinic] -42
 RHS1 axisVdifLB[c,User,Location] -42 axisVdifLB[c,User,Therapist] -42
 RHS1 axisVdifLB[c,User,Instructor] -42 axisVdifLB[c,CourseSession,Clinic] -42
 RHS1 axisVdifLB[c,CourseSession,Location] -42 axisVdifLB[c,CourseSession,Therapist] -42
 RHS1 axisVdifLB[c,CourseSession,Instructor] -42 axisVdifLB[c,Clinic,Location] -42
 RHS1 axisVdifLB[c,Clinic,Therapist] -42 axisVdifLB[c,Clinic,Instructor] -42
 RHS1 axisVdifLB[c,Location,Therapist] -42 axisVdifLB[c,Location,Instructor] -42
 RHS1 axisVdifLB[c,Therapist,Instructor] -42 axisVdifRT[r,Client,CertificateTemplate] -42
 RHS1 axisVdifRT[r,Client,Course] -42 axisVdifRT[r,Client,Role] -42
 RHS1 axisVdifRT[r,Client,User] -42 axisVdifRT[r,Client,CourseSession] -42
 RHS1 axisVdifRT[r,Client,Clinic] -42 axisVdifRT[r,Client,Location] -42
 RHS1 axisVdifRT[r,Client,Therapist] -42 axisVdifRT[r,Client,Instructor] -42
 RHS1 axisVdifRT[r,CertificateTemplate,Course] -42 axisVdifRT[r,CertificateTemplate,Role] -42
 RHS1 axisVdifRT[r,CertificateTemplate,User] -42 axisVdifRT[r,CertificateTemplate,CourseSession] -42
 RHS1 axisVdifRT[r,CertificateTemplate,Clinic] -42 axisVdifRT[r,CertificateTemplate,Location] -42
 RHS1 axisVdifRT[r,CertificateTemplate,Therapist] -42 axisVdifRT[r,CertificateTemplate,Instructor] -42
 RHS1 axisVdifRT[r,Course,Role] -42 axisVdifRT[r,Course,User] -42
 RHS1 axisVdifRT[r,Course,CourseSession] -42 axisVdifRT[r,Course,Clinic] -42
 RHS1 axisVdifRT[r,Course,Location] -42 axisVdifRT[r,Course,Therapist] -42
 RHS1 axisVdifRT[r,Course,Instructor] -42 axisVdifRT[r,Role,User] -42
 RHS1 axisVdifRT[r,Role,CourseSession] -42 axisVdifRT[r,Role,Clinic] -42
 RHS1 axisVdifRT[r,Role,Location] -42 axisVdifRT[r,Role,Therapist] -42
 RHS1 axisVdifRT[r,Role,Instructor] -42 axisVdifRT[r,User,CourseSession] -42
 RHS1 axisVdifRT[r,User,Clinic] -42 axisVdifRT[r,User,Location] -42
 RHS1 axisVdifRT[r,User,Therapist] -42 axisVdifRT[r,User,Instructor] -42
 RHS1 axisVdifRT[r,CourseSession,Clinic] -42 axisVdifRT[r,CourseSession,Location] -42
 RHS1 axisVdifRT[r,CourseSession,Therapist] -42 axisVdifRT[r,CourseSession,Instructor] -42
 RHS1 axisVdifRT[r,Clinic,Location] -42 axisVdifRT[r,Clinic,Therapist] -42
 RHS1 axisVdifRT[r,Clinic,Instructor] -42 axisVdifRT[r,Location,Therapist] -42
 RHS1 axisVdifRT[r,Location,Instructor] -42 axisVdifRT[r,Therapist,Instructor] -42
 RHS1 axisVdifRT[c,Client,CertificateTemplate] -42 axisVdifRT[c,Client,Course] -42
 RHS1 axisVdifRT[c,Client,Role] -42 axisVdifRT[c,Client,User] -42
 RHS1 axisVdifRT[c,Client,CourseSession] -42 axisVdifRT[c,Client,Clinic] -42
 RHS1 axisVdifRT[c,Client,Location] -42 axisVdifRT[c,Client,Therapist] -42
 RHS1 axisVdifRT[c,Client,Instructor] -42 axisVdifRT[c,CertificateTemplate,Course] -42
 RHS1 axisVdifRT[c,CertificateTemplate,Role] -42 axisVdifRT[c,CertificateTemplate,User] -42
 RHS1 axisVdifRT[c,CertificateTemplate,CourseSession] -42 axisVdifRT[c,CertificateTemplate,Clinic] -42
 RHS1 axisVdifRT[c,CertificateTemplate,Location] -42 axisVdifRT[c,CertificateTemplate,Therapist] -42
 RHS1 axisVdifRT[c,CertificateTemplate,Instructor] -42 axisVdifRT[c,Course,Role] -42
 RHS1 axisVdifRT[c,Course,User] -42 axisVdifRT[c,Course,CourseSession] -42
 RHS1 axisVdifRT[c,Course,Clinic] -42 axisVdifRT[c,Course,Location] -42
 RHS1 axisVdifRT[c,Course,Therapist] -42 axisVdifRT[c,Course,Instructor] -42
 RHS1 axisVdifRT[c,Role,User] -42 axisVdifRT[c,Role,CourseSession] -42
 RHS1 axisVdifRT[c,Role,Clinic] -42 axisVdifRT[c,Role,Location] -42
 RHS1 axisVdifRT[c,Role,Therapist] -42 axisVdifRT[c,Role,Instructor] -42
 RHS1 axisVdifRT[c,User,CourseSession] -42 axisVdifRT[c,User,Clinic] -42
 RHS1 axisVdifRT[c,User,Location] -42 axisVdifRT[c,User,Therapist] -42
 RHS1 axisVdifRT[c,User,Instructor] -42 axisVdifRT[c,CourseSession,Clinic] -42
 RHS1 axisVdifRT[c,CourseSession,Location] -42 axisVdifRT[c,CourseSession,Therapist] -42
 RHS1 axisVdifRT[c,CourseSession,Instructor] -42 axisVdifRT[c,Clinic,Location] -42
 RHS1 axisVdifRT[c,Clinic,Therapist] -42 axisVdifRT[c,Clinic,Instructor] -42
 RHS1 axisVdifRT[c,Location,Therapist] -42 axisVdifRT[c,Location,Instructor] -42
 RHS1 axisVdifRT[c,Therapist,Instructor] -42 compDistAxis1[r,Client,CertificateTemplate] 1
 RHS1 compDistAxis1[r,Client,Instructor] -0.5 compDistAxis1[r,Client,User] 1
 RHS1 compDistAxis1[r,CertificateTemplate,CourseSession] -1 compDistAxis1[r,Course,CourseSession] -1.5
 RHS1 compDistAxis1[r,CourseSession,Instructor] -0.5 compDistAxis1[r,Instructor,Location] 0.5
 RHS1 compDistAxis1[r,Clinic,Location] -0.5 compDistAxis1[r,Clinic,Therapist] -1
 RHS1 compDistAxis1[r,Location,Therapist] -0.5 compDistAxis1[r,Therapist,CourseSession] 0.5
 RHS1 compDistAxis1[r,Role,User] -0.5 compDistAxis1[r,User,Instructor] -1.5
 RHS1 compDistAxis1[r,User,Therapist] -1.5 compDistAxis1[c,Client,CertificateTemplate] -0.5
 RHS1 compDistAxis1[c,Client,CourseSession] -1.5 compDistAxis1[c,Client,Instructor] -0.5
 RHS1 compDistAxis1[c,CertificateTemplate,CourseSession] -1 compDistAxis1[c,Course,CourseSession] -2
 RHS1 compDistAxis1[c,CourseSession,Instructor] 1 compDistAxis1[c,Instructor,Location] 0.5
 RHS1 compDistAxis1[c,Location,CourseSession] -1.5 compDistAxis1[c,Therapist,CourseSession] -1.5
 RHS1 compDistAxis1[c,Role,User] -0.5 compDistAxis1[c,User,Instructor] -0.5
 RHS1 compDistAxis2[r,Client,CertificateTemplate] -1 compDistAxis2[r,Client,Instructor] 0.5
 RHS1 compDistAxis2[r,Client,User] -1 compDistAxis2[r,CertificateTemplate,CourseSession] 1
 RHS1 compDistAxis2[r,Course,CourseSession] 1.5 compDistAxis2[r,CourseSession,Instructor] 0.5
 RHS1 compDistAxis2[r,Instructor,Location] -0.5 compDistAxis2[r,Clinic,Location] 0.5
 RHS1 compDistAxis2[r,Clinic,Therapist] 1 compDistAxis2[r,Location,Therapist] 0.5
 RHS1 compDistAxis2[r,Therapist,CourseSession] -0.5 compDistAxis2[r,Role,User] 0.5
 RHS1 compDistAxis2[r,User,Instructor] 1.5 compDistAxis2[r,User,Therapist] 1.5
 RHS1 compDistAxis2[c,Client,CertificateTemplate] 0.5 compDistAxis2[c,Client,CourseSession] 1.5
 RHS1 compDistAxis2[c,Client,Instructor] 0.5 compDistAxis2[c,CertificateTemplate,CourseSession] 1
 RHS1 compDistAxis2[c,Course,CourseSession] 2 compDistAxis2[c,CourseSession,Instructor] -1
 RHS1 compDistAxis2[c,Instructor,Location] -0.5 compDistAxis2[c,Location,CourseSession] 1.5
 RHS1 compDistAxis2[c,Therapist,CourseSession] 1.5 compDistAxis2[c,Role,User] 0.5
 RHS1 compDistAxis2[c,User,Instructor] 0.5 DistSumLB[Client,CertificateTemplate] 8
 RHS1 DistSumLB[Client,CourseSession] 9 DistSumLB[Client,Instructor] 9.5
 RHS1 DistSumLB[Client,Location] 9 DistSumLB[Client,User] 8
 RHS1 DistSumLB[CertificateTemplate,CourseSession] 8 DistSumLB[Course,CourseSession] 7.5
 RHS1 DistSumLB[CourseSession,Instructor] 9.5 DistSumLB[Instructor,Location] 9.5
 RHS1 DistSumLB[Clinic,Location] 8.5 DistSumLB[Clinic,Therapist] 9
 RHS1 DistSumLB[Location,Therapist] 9.5 DistSumLB[Location,CourseSession] 9
 RHS1 DistSumLB[Therapist,CourseSession] 9.5 DistSumLB[Role,User] 6.5
 RHS1 DistSumLB[User,Instructor] 8.5 DistSumLB[User,Therapist] 8.5
 RHS1 CenterDistSumLB 30 DistAxisLB1[r,Client,CertificateTemplate] 8
 RHS1 DistAxisLB1[r,Client,CourseSession] 9 DistAxisLB1[r,Client,Instructor] 9.5
 RHS1 DistAxisLB1[r,Client,Location] 9 DistAxisLB1[r,Client,User] 8
 RHS1 DistAxisLB1[r,CertificateTemplate,CourseSession] 8 DistAxisLB1[r,Course,CourseSession] 7.5
 RHS1 DistAxisLB1[r,CourseSession,Instructor] 9.5 DistAxisLB1[r,Instructor,Location] 9.5
 RHS1 DistAxisLB1[r,Clinic,Location] 8.5 DistAxisLB1[r,Clinic,Therapist] 9
 RHS1 DistAxisLB1[r,Location,Therapist] 9.5 DistAxisLB1[r,Location,CourseSession] 9
 RHS1 DistAxisLB1[r,Therapist,CourseSession] 9.5 DistAxisLB1[r,Role,User] 6.5
 RHS1 DistAxisLB1[r,User,Instructor] 8.5 DistAxisLB1[r,User,Therapist] 8.5
 RHS1 DistAxisLB1[c,Client,CertificateTemplate] 11.5 DistAxisLB1[c,Client,CourseSession] 12.5
 RHS1 DistAxisLB1[c,Client,Instructor] 11.5 DistAxisLB1[c,Client,Location] 11
 RHS1 DistAxisLB1[c,Client,User] 11 DistAxisLB1[c,CertificateTemplate,CourseSession] 13
 RHS1 DistAxisLB1[c,Course,CourseSession] 12 DistAxisLB1[c,CourseSession,Instructor] 13
 RHS1 DistAxisLB1[c,Instructor,Location] 11.5 DistAxisLB1[c,Clinic,Location] 11
 RHS1 DistAxisLB1[c,Clinic,Therapist] 11 DistAxisLB1[c,Location,Therapist] 11
 RHS1 DistAxisLB1[c,Location,CourseSession] 12.5 DistAxisLB1[c,Therapist,CourseSession] 12.5
 RHS1 DistAxisLB1[c,Role,User] 10.5 DistAxisLB1[c,User,Instructor] 11.5
 RHS1 DistAxisLB1[c,User,Therapist] 11 triangle1[Client,CertificateTemplate,Course,r] 2
 RHS1 triangle1[Client,CertificateTemplate,Course,c] 2 triangle1[Client,CertificateTemplate,Role,r] 2
 RHS1 triangle1[Client,CertificateTemplate,Role,c] 2 triangle1[Client,CertificateTemplate,User,r] 2
 RHS1 triangle1[Client,CertificateTemplate,User,c] 2 triangle1[Client,CertificateTemplate,CourseSession,r] 2
 RHS1 triangle1[Client,CertificateTemplate,CourseSession,c] 2 triangle1[Client,CertificateTemplate,Clinic,r] 2
 RHS1 triangle1[Client,CertificateTemplate,Clinic,c] 2 triangle1[Client,CertificateTemplate,Location,r] 2
 RHS1 triangle1[Client,CertificateTemplate,Location,c] 2 triangle1[Client,CertificateTemplate,Therapist,r] 2
 RHS1 triangle1[Client,CertificateTemplate,Therapist,c] 2 triangle1[Client,CertificateTemplate,Instructor,r] 2
 RHS1 triangle1[Client,CertificateTemplate,Instructor,c] 2 triangle1[Client,Course,Role,r] 2
 RHS1 triangle1[Client,Course,Role,c] 2 triangle1[Client,Course,User,r] 2
 RHS1 triangle1[Client,Course,User,c] 2 triangle1[Client,Course,CourseSession,r] 2
 RHS1 triangle1[Client,Course,CourseSession,c] 2 triangle1[Client,Course,Clinic,r] 2
 RHS1 triangle1[Client,Course,Clinic,c] 2 triangle1[Client,Course,Location,r] 2
 RHS1 triangle1[Client,Course,Location,c] 2 triangle1[Client,Course,Therapist,r] 2
 RHS1 triangle1[Client,Course,Therapist,c] 2 triangle1[Client,Course,Instructor,r] 2
 RHS1 triangle1[Client,Course,Instructor,c] 2 triangle1[Client,Role,User,r] 2
 RHS1 triangle1[Client,Role,User,c] 2 triangle1[Client,Role,CourseSession,r] 2
 RHS1 triangle1[Client,Role,CourseSession,c] 2 triangle1[Client,Role,Clinic,r] 2
 RHS1 triangle1[Client,Role,Clinic,c] 2 triangle1[Client,Role,Location,r] 2
 RHS1 triangle1[Client,Role,Location,c] 2 triangle1[Client,Role,Therapist,r] 2
 RHS1 triangle1[Client,Role,Therapist,c] 2 triangle1[Client,Role,Instructor,r] 2
 RHS1 triangle1[Client,Role,Instructor,c] 2 triangle1[Client,User,CourseSession,r] 2
 RHS1 triangle1[Client,User,CourseSession,c] 2 triangle1[Client,User,Clinic,r] 2
 RHS1 triangle1[Client,User,Clinic,c] 2 triangle1[Client,User,Location,r] 2
 RHS1 triangle1[Client,User,Location,c] 2 triangle1[Client,User,Therapist,r] 2
 RHS1 triangle1[Client,User,Therapist,c] 2 triangle1[Client,User,Instructor,r] 2
 RHS1 triangle1[Client,User,Instructor,c] 2 triangle1[Client,CourseSession,Clinic,r] 2
 RHS1 triangle1[Client,CourseSession,Clinic,c] 2 triangle1[Client,CourseSession,Location,r] 2
 RHS1 triangle1[Client,CourseSession,Location,c] 2 triangle1[Client,CourseSession,Therapist,r] 2
 RHS1 triangle1[Client,CourseSession,Therapist,c] 2 triangle1[Client,CourseSession,Instructor,r] 2
 RHS1 triangle1[Client,CourseSession,Instructor,c] 2 triangle1[Client,Clinic,Location,r] 2
 RHS1 triangle1[Client,Clinic,Location,c] 2 triangle1[Client,Clinic,Therapist,r] 2
 RHS1 triangle1[Client,Clinic,Therapist,c] 2 triangle1[Client,Clinic,Instructor,r] 2
 RHS1 triangle1[Client,Clinic,Instructor,c] 2 triangle1[Client,Location,Therapist,r] 2
 RHS1 triangle1[Client,Location,Therapist,c] 2 triangle1[Client,Location,Instructor,r] 2
 RHS1 triangle1[Client,Location,Instructor,c] 2 triangle1[Client,Therapist,Instructor,r] 2
 RHS1 triangle1[Client,Therapist,Instructor,c] 2 triangle1[CertificateTemplate,Course,Role,r] 2
 RHS1 triangle1[CertificateTemplate,Course,Role,c] 2 triangle1[CertificateTemplate,Course,User,r] 2
 RHS1 triangle1[CertificateTemplate,Course,User,c] 2 triangle1[CertificateTemplate,Course,CourseSession,r] 2
 RHS1 triangle1[CertificateTemplate,Course,CourseSession,c] 2 triangle1[CertificateTemplate,Course,Clinic,r] 2
 RHS1 triangle1[CertificateTemplate,Course,Clinic,c] 2 triangle1[CertificateTemplate,Course,Location,r] 2
 RHS1 triangle1[CertificateTemplate,Course,Location,c] 2 triangle1[CertificateTemplate,Course,Therapist,r] 2
 RHS1 triangle1[CertificateTemplate,Course,Therapist,c] 2 triangle1[CertificateTemplate,Course,Instructor,r] 2
 RHS1 triangle1[CertificateTemplate,Course,Instructor,c] 2 triangle1[CertificateTemplate,Role,User,r] 2
 RHS1 triangle1[CertificateTemplate,Role,User,c] 2 triangle1[CertificateTemplate,Role,CourseSession,r] 2
 RHS1 triangle1[CertificateTemplate,Role,CourseSession,c] 2 triangle1[CertificateTemplate,Role,Clinic,r] 2
 RHS1 triangle1[CertificateTemplate,Role,Clinic,c] 2 triangle1[CertificateTemplate,Role,Location,r] 2
 RHS1 triangle1[CertificateTemplate,Role,Location,c] 2 triangle1[CertificateTemplate,Role,Therapist,r] 2
 RHS1 triangle1[CertificateTemplate,Role,Therapist,c] 2 triangle1[CertificateTemplate,Role,Instructor,r] 2
 RHS1 triangle1[CertificateTemplate,Role,Instructor,c] 2 triangle1[CertificateTemplate,User,CourseSession,r] 2
 RHS1 triangle1[CertificateTemplate,User,CourseSession,c] 2 triangle1[CertificateTemplate,User,Clinic,r] 2
 RHS1 triangle1[CertificateTemplate,User,Clinic,c] 2 triangle1[CertificateTemplate,User,Location,r] 2
 RHS1 triangle1[CertificateTemplate,User,Location,c] 2 triangle1[CertificateTemplate,User,Therapist,r] 2
 RHS1 triangle1[CertificateTemplate,User,Therapist,c] 2 triangle1[CertificateTemplate,User,Instructor,r] 2
 RHS1 triangle1[CertificateTemplate,User,Instructor,c] 2 triangle1[CertificateTemplate,CourseSession,Clinic,r] 2
 RHS1 triangle1[CertificateTemplate,CourseSession,Clinic,c] 2 triangle1[CertificateTemplate,CourseSession,Location,r] 2
 RHS1 triangle1[CertificateTemplate,CourseSession,Location,c] 2 triangle1[CertificateTemplate,CourseSession,Therapist,r] 2
 RHS1 triangle1[CertificateTemplate,CourseSession,Therapist,c] 2 triangle1[CertificateTemplate,CourseSession,Instructor,r] 2
 RHS1 triangle1[CertificateTemplate,CourseSession,Instructor,c] 2 triangle1[CertificateTemplate,Clinic,Location,r] 2
 RHS1 triangle1[CertificateTemplate,Clinic,Location,c] 2 triangle1[CertificateTemplate,Clinic,Therapist,r] 2
 RHS1 triangle1[CertificateTemplate,Clinic,Therapist,c] 2 triangle1[CertificateTemplate,Clinic,Instructor,r] 2
 RHS1 triangle1[CertificateTemplate,Clinic,Instructor,c] 2 triangle1[CertificateTemplate,Location,Therapist,r] 2
 RHS1 triangle1[CertificateTemplate,Location,Therapist,c] 2 triangle1[CertificateTemplate,Location,Instructor,r] 2
 RHS1 triangle1[CertificateTemplate,Location,Instructor,c] 2 triangle1[CertificateTemplate,Therapist,Instructor,r] 2
 RHS1 triangle1[CertificateTemplate,Therapist,Instructor,c] 2 triangle1[Course,Role,User,r] 2
 RHS1 triangle1[Course,Role,User,c] 2 triangle1[Course,Role,CourseSession,r] 2
 RHS1 triangle1[Course,Role,CourseSession,c] 2 triangle1[Course,Role,Clinic,r] 2
 RHS1 triangle1[Course,Role,Clinic,c] 2 triangle1[Course,Role,Location,r] 2
 RHS1 triangle1[Course,Role,Location,c] 2 triangle1[Course,Role,Therapist,r] 2
 RHS1 triangle1[Course,Role,Therapist,c] 2 triangle1[Course,Role,Instructor,r] 2
 RHS1 triangle1[Course,Role,Instructor,c] 2 triangle1[Course,User,CourseSession,r] 2
 RHS1 triangle1[Course,User,CourseSession,c] 2 triangle1[Course,User,Clinic,r] 2
 RHS1 triangle1[Course,User,Clinic,c] 2 triangle1[Course,User,Location,r] 2
 RHS1 triangle1[Course,User,Location,c] 2 triangle1[Course,User,Therapist,r] 2
 RHS1 triangle1[Course,User,Therapist,c] 2 triangle1[Course,User,Instructor,r] 2
 RHS1 triangle1[Course,User,Instructor,c] 2 triangle1[Course,CourseSession,Clinic,r] 2
 RHS1 triangle1[Course,CourseSession,Clinic,c] 2 triangle1[Course,CourseSession,Location,r] 2
 RHS1 triangle1[Course,CourseSession,Location,c] 2 triangle1[Course,CourseSession,Therapist,r] 2
 RHS1 triangle1[Course,CourseSession,Therapist,c] 2 triangle1[Course,CourseSession,Instructor,r] 2
 RHS1 triangle1[Course,CourseSession,Instructor,c] 2 triangle1[Course,Clinic,Location,r] 2
 RHS1 triangle1[Course,Clinic,Location,c] 2 triangle1[Course,Clinic,Therapist,r] 2
 RHS1 triangle1[Course,Clinic,Therapist,c] 2 triangle1[Course,Clinic,Instructor,r] 2
 RHS1 triangle1[Course,Clinic,Instructor,c] 2 triangle1[Course,Location,Therapist,r] 2
 RHS1 triangle1[Course,Location,Therapist,c] 2 triangle1[Course,Location,Instructor,r] 2
 RHS1 triangle1[Course,Location,Instructor,c] 2 triangle1[Course,Therapist,Instructor,r] 2
 RHS1 triangle1[Course,Therapist,Instructor,c] 2 triangle1[Role,User,CourseSession,r] 2
 RHS1 triangle1[Role,User,CourseSession,c] 2 triangle1[Role,User,Clinic,r] 2
 RHS1 triangle1[Role,User,Clinic,c] 2 triangle1[Role,User,Location,r] 2
 RHS1 triangle1[Role,User,Location,c] 2 triangle1[Role,User,Therapist,r] 2
 RHS1 triangle1[Role,User,Therapist,c] 2 triangle1[Role,User,Instructor,r] 2
 RHS1 triangle1[Role,User,Instructor,c] 2 triangle1[Role,CourseSession,Clinic,r] 2
 RHS1 triangle1[Role,CourseSession,Clinic,c] 2 triangle1[Role,CourseSession,Location,r] 2
 RHS1 triangle1[Role,CourseSession,Location,c] 2 triangle1[Role,CourseSession,Therapist,r] 2
 RHS1 triangle1[Role,CourseSession,Therapist,c] 2 triangle1[Role,CourseSession,Instructor,r] 2
 RHS1 triangle1[Role,CourseSession,Instructor,c] 2 triangle1[Role,Clinic,Location,r] 2
 RHS1 triangle1[Role,Clinic,Location,c] 2 triangle1[Role,Clinic,Therapist,r] 2
 RHS1 triangle1[Role,Clinic,Therapist,c] 2 triangle1[Role,Clinic,Instructor,r] 2
 RHS1 triangle1[Role,Clinic,Instructor,c] 2 triangle1[Role,Location,Therapist,r] 2
 RHS1 triangle1[Role,Location,Therapist,c] 2 triangle1[Role,Location,Instructor,r] 2
 RHS1 triangle1[Role,Location,Instructor,c] 2 triangle1[Role,Therapist,Instructor,r] 2
 RHS1 triangle1[Role,Therapist,Instructor,c] 2 triangle1[User,CourseSession,Clinic,r] 2
 RHS1 triangle1[User,CourseSession,Clinic,c] 2 triangle1[User,CourseSession,Location,r] 2
 RHS1 triangle1[User,CourseSession,Location,c] 2 triangle1[User,CourseSession,Therapist,r] 2
 RHS1 triangle1[User,CourseSession,Therapist,c] 2 triangle1[User,CourseSession,Instructor,r] 2
 RHS1 triangle1[User,CourseSession,Instructor,c] 2 triangle1[User,Clinic,Location,r] 2
 RHS1 triangle1[User,Clinic,Location,c] 2 triangle1[User,Clinic,Therapist,r] 2
 RHS1 triangle1[User,Clinic,Therapist,c] 2 triangle1[User,Clinic,Instructor,r] 2
 RHS1 triangle1[User,Clinic,Instructor,c] 2 triangle1[User,Location,Therapist,r] 2
 RHS1 triangle1[User,Location,Therapist,c] 2 triangle1[User,Location,Instructor,r] 2
 RHS1 triangle1[User,Location,Instructor,c] 2 triangle1[User,Therapist,Instructor,r] 2
 RHS1 triangle1[User,Therapist,Instructor,c] 2 triangle1[CourseSession,Clinic,Location,r] 2
 RHS1 triangle1[CourseSession,Clinic,Location,c] 2 triangle1[CourseSession,Clinic,Therapist,r] 2
 RHS1 triangle1[CourseSession,Clinic,Therapist,c] 2 triangle1[CourseSession,Clinic,Instructor,r] 2
 RHS1 triangle1[CourseSession,Clinic,Instructor,c] 2 triangle1[CourseSession,Location,Therapist,r] 2
 RHS1 triangle1[CourseSession,Location,Therapist,c] 2 triangle1[CourseSession,Location,Instructor,r] 2
 RHS1 triangle1[CourseSession,Location,Instructor,c] 2 triangle1[CourseSession,Therapist,Instructor,r] 2
 RHS1 triangle1[CourseSession,Therapist,Instructor,c] 2 triangle1[Clinic,Location,Therapist,r] 2
 RHS1 triangle1[Clinic,Location,Therapist,c] 2 triangle1[Clinic,Location,Instructor,r] 2
 RHS1 triangle1[Clinic,Location,Instructor,c] 2 triangle1[Clinic,Therapist,Instructor,r] 2
 RHS1 triangle1[Clinic,Therapist,Instructor,c] 2 triangle1[Location,Therapist,Instructor,r] 2
 RHS1 triangle1[Location,Therapist,Instructor,c] 2 triangle2[Client,CertificateTemplate,Course,r] 2
 RHS1 triangle2[Client,CertificateTemplate,Course,c] 2 triangle2[Client,CertificateTemplate,Role,r] 2
 RHS1 triangle2[Client,CertificateTemplate,Role,c] 2 triangle2[Client,CertificateTemplate,User,r] 2
 RHS1 triangle2[Client,CertificateTemplate,User,c] 2 triangle2[Client,CertificateTemplate,CourseSession,r] 2
 RHS1 triangle2[Client,CertificateTemplate,CourseSession,c] 2 triangle2[Client,CertificateTemplate,Clinic,r] 2
 RHS1 triangle2[Client,CertificateTemplate,Clinic,c] 2 triangle2[Client,CertificateTemplate,Location,r] 2
 RHS1 triangle2[Client,CertificateTemplate,Location,c] 2 triangle2[Client,CertificateTemplate,Therapist,r] 2
 RHS1 triangle2[Client,CertificateTemplate,Therapist,c] 2 triangle2[Client,CertificateTemplate,Instructor,r] 2
 RHS1 triangle2[Client,CertificateTemplate,Instructor,c] 2 triangle2[Client,Course,Role,r] 2
 RHS1 triangle2[Client,Course,Role,c] 2 triangle2[Client,Course,User,r] 2
 RHS1 triangle2[Client,Course,User,c] 2 triangle2[Client,Course,CourseSession,r] 2
 RHS1 triangle2[Client,Course,CourseSession,c] 2 triangle2[Client,Course,Clinic,r] 2
 RHS1 triangle2[Client,Course,Clinic,c] 2 triangle2[Client,Course,Location,r] 2
 RHS1 triangle2[Client,Course,Location,c] 2 triangle2[Client,Course,Therapist,r] 2
 RHS1 triangle2[Client,Course,Therapist,c] 2 triangle2[Client,Course,Instructor,r] 2
 RHS1 triangle2[Client,Course,Instructor,c] 2 triangle2[Client,Role,User,r] 2
 RHS1 triangle2[Client,Role,User,c] 2 triangle2[Client,Role,CourseSession,r] 2
 RHS1 triangle2[Client,Role,CourseSession,c] 2 triangle2[Client,Role,Clinic,r] 2
 RHS1 triangle2[Client,Role,Clinic,c] 2 triangle2[Client,Role,Location,r] 2
 RHS1 triangle2[Client,Role,Location,c] 2 triangle2[Client,Role,Therapist,r] 2
 RHS1 triangle2[Client,Role,Therapist,c] 2 triangle2[Client,Role,Instructor,r] 2
 RHS1 triangle2[Client,Role,Instructor,c] 2 triangle2[Client,User,CourseSession,r] 2
 RHS1 triangle2[Client,User,CourseSession,c] 2 triangle2[Client,User,Clinic,r] 2
 RHS1 triangle2[Client,User,Clinic,c] 2 triangle2[Client,User,Location,r] 2
 RHS1 triangle2[Client,User,Location,c] 2 triangle2[Client,User,Therapist,r] 2
 RHS1 triangle2[Client,User,Therapist,c] 2 triangle2[Client,User,Instructor,r] 2
 RHS1 triangle2[Client,User,Instructor,c] 2 triangle2[Client,CourseSession,Clinic,r] 2
 RHS1 triangle2[Client,CourseSession,Clinic,c] 2 triangle2[Client,CourseSession,Location,r] 2
 RHS1 triangle2[Client,CourseSession,Location,c] 2 triangle2[Client,CourseSession,Therapist,r] 2
 RHS1 triangle2[Client,CourseSession,Therapist,c] 2 triangle2[Client,CourseSession,Instructor,r] 2
 RHS1 triangle2[Client,CourseSession,Instructor,c] 2 triangle2[Client,Clinic,Location,r] 2
 RHS1 triangle2[Client,Clinic,Location,c] 2 triangle2[Client,Clinic,Therapist,r] 2
 RHS1 triangle2[Client,Clinic,Therapist,c] 2 triangle2[Client,Clinic,Instructor,r] 2
 RHS1 triangle2[Client,Clinic,Instructor,c] 2 triangle2[Client,Location,Therapist,r] 2
 RHS1 triangle2[Client,Location,Therapist,c] 2 triangle2[Client,Location,Instructor,r] 2
 RHS1 triangle2[Client,Location,Instructor,c] 2 triangle2[Client,Therapist,Instructor,r] 2
 RHS1 triangle2[Client,Therapist,Instructor,c] 2 triangle2[CertificateTemplate,Course,Role,r] 2
 RHS1 triangle2[CertificateTemplate,Course,Role,c] 2 triangle2[CertificateTemplate,Course,User,r] 2
 RHS1 triangle2[CertificateTemplate,Course,User,c] 2 triangle2[CertificateTemplate,Course,CourseSession,r] 2
 RHS1 triangle2[CertificateTemplate,Course,CourseSession,c] 2 triangle2[CertificateTemplate,Course,Clinic,r] 2
 RHS1 triangle2[CertificateTemplate,Course,Clinic,c] 2 triangle2[CertificateTemplate,Course,Location,r] 2
 RHS1 triangle2[CertificateTemplate,Course,Location,c] 2 triangle2[CertificateTemplate,Course,Therapist,r] 2
 RHS1 triangle2[CertificateTemplate,Course,Therapist,c] 2 triangle2[CertificateTemplate,Course,Instructor,r] 2
 RHS1 triangle2[CertificateTemplate,Course,Instructor,c] 2 triangle2[CertificateTemplate,Role,User,r] 2
 RHS1 triangle2[CertificateTemplate,Role,User,c] 2 triangle2[CertificateTemplate,Role,CourseSession,r] 2
 RHS1 triangle2[CertificateTemplate,Role,CourseSession,c] 2 triangle2[CertificateTemplate,Role,Clinic,r] 2
 RHS1 triangle2[CertificateTemplate,Role,Clinic,c] 2 triangle2[CertificateTemplate,Role,Location,r] 2
 RHS1 triangle2[CertificateTemplate,Role,Location,c] 2 triangle2[CertificateTemplate,Role,Therapist,r] 2
 RHS1 triangle2[CertificateTemplate,Role,Therapist,c] 2 triangle2[CertificateTemplate,Role,Instructor,r] 2
 RHS1 triangle2[CertificateTemplate,Role,Instructor,c] 2 triangle2[CertificateTemplate,User,CourseSession,r] 2
 RHS1 triangle2[CertificateTemplate,User,CourseSession,c] 2 triangle2[CertificateTemplate,User,Clinic,r] 2
 RHS1 triangle2[CertificateTemplate,User,Clinic,c] 2 triangle2[CertificateTemplate,User,Location,r] 2
 RHS1 triangle2[CertificateTemplate,User,Location,c] 2 triangle2[CertificateTemplate,User,Therapist,r] 2
 RHS1 triangle2[CertificateTemplate,User,Therapist,c] 2 triangle2[CertificateTemplate,User,Instructor,r] 2
 RHS1 triangle2[CertificateTemplate,User,Instructor,c] 2 triangle2[CertificateTemplate,CourseSession,Clinic,r] 2
 RHS1 triangle2[CertificateTemplate,CourseSession,Clinic,c] 2 triangle2[CertificateTemplate,CourseSession,Location,r] 2
 RHS1 triangle2[CertificateTemplate,CourseSession,Location,c] 2 triangle2[CertificateTemplate,CourseSession,Therapist,r] 2
 RHS1 triangle2[CertificateTemplate,CourseSession,Therapist,c] 2 triangle2[CertificateTemplate,CourseSession,Instructor,r] 2
 RHS1 triangle2[CertificateTemplate,CourseSession,Instructor,c] 2 triangle2[CertificateTemplate,Clinic,Location,r] 2
 RHS1 triangle2[CertificateTemplate,Clinic,Location,c] 2 triangle2[CertificateTemplate,Clinic,Therapist,r] 2
 RHS1 triangle2[CertificateTemplate,Clinic,Therapist,c] 2 triangle2[CertificateTemplate,Clinic,Instructor,r] 2
 RHS1 triangle2[CertificateTemplate,Clinic,Instructor,c] 2 triangle2[CertificateTemplate,Location,Therapist,r] 2
 RHS1 triangle2[CertificateTemplate,Location,Therapist,c] 2 triangle2[CertificateTemplate,Location,Instructor,r] 2
 RHS1 triangle2[CertificateTemplate,Location,Instructor,c] 2 triangle2[CertificateTemplate,Therapist,Instructor,r] 2
 RHS1 triangle2[CertificateTemplate,Therapist,Instructor,c] 2 triangle2[Course,Role,User,r] 2
 RHS1 triangle2[Course,Role,User,c] 2 triangle2[Course,Role,CourseSession,r] 2
 RHS1 triangle2[Course,Role,CourseSession,c] 2 triangle2[Course,Role,Clinic,r] 2
 RHS1 triangle2[Course,Role,Clinic,c] 2 triangle2[Course,Role,Location,r] 2
 RHS1 triangle2[Course,Role,Location,c] 2 triangle2[Course,Role,Therapist,r] 2
 RHS1 triangle2[Course,Role,Therapist,c] 2 triangle2[Course,Role,Instructor,r] 2
 RHS1 triangle2[Course,Role,Instructor,c] 2 triangle2[Course,User,CourseSession,r] 2
 RHS1 triangle2[Course,User,CourseSession,c] 2 triangle2[Course,User,Clinic,r] 2
 RHS1 triangle2[Course,User,Clinic,c] 2 triangle2[Course,User,Location,r] 2
 RHS1 triangle2[Course,User,Location,c] 2 triangle2[Course,User,Therapist,r] 2
 RHS1 triangle2[Course,User,Therapist,c] 2 triangle2[Course,User,Instructor,r] 2
 RHS1 triangle2[Course,User,Instructor,c] 2 triangle2[Course,CourseSession,Clinic,r] 2
 RHS1 triangle2[Course,CourseSession,Clinic,c] 2 triangle2[Course,CourseSession,Location,r] 2
 RHS1 triangle2[Course,CourseSession,Location,c] 2 triangle2[Course,CourseSession,Therapist,r] 2
 RHS1 triangle2[Course,CourseSession,Therapist,c] 2 triangle2[Course,CourseSession,Instructor,r] 2
 RHS1 triangle2[Course,CourseSession,Instructor,c] 2 triangle2[Course,Clinic,Location,r] 2
 RHS1 triangle2[Course,Clinic,Location,c] 2 triangle2[Course,Clinic,Therapist,r] 2
 RHS1 triangle2[Course,Clinic,Therapist,c] 2 triangle2[Course,Clinic,Instructor,r] 2
 RHS1 triangle2[Course,Clinic,Instructor,c] 2 triangle2[Course,Location,Therapist,r] 2
 RHS1 triangle2[Course,Location,Therapist,c] 2 triangle2[Course,Location,Instructor,r] 2
 RHS1 triangle2[Course,Location,Instructor,c] 2 triangle2[Course,Therapist,Instructor,r] 2
 RHS1 triangle2[Course,Therapist,Instructor,c] 2 triangle2[Role,User,CourseSession,r] 2
 RHS1 triangle2[Role,User,CourseSession,c] 2 triangle2[Role,User,Clinic,r] 2
 RHS1 triangle2[Role,User,Clinic,c] 2 triangle2[Role,User,Location,r] 2
 RHS1 triangle2[Role,User,Location,c] 2 triangle2[Role,User,Therapist,r] 2
 RHS1 triangle2[Role,User,Therapist,c] 2 triangle2[Role,User,Instructor,r] 2
 RHS1 triangle2[Role,User,Instructor,c] 2 triangle2[Role,CourseSession,Clinic,r] 2
 RHS1 triangle2[Role,CourseSession,Clinic,c] 2 triangle2[Role,CourseSession,Location,r] 2
 RHS1 triangle2[Role,CourseSession,Location,c] 2 triangle2[Role,CourseSession,Therapist,r] 2
 RHS1 triangle2[Role,CourseSession,Therapist,c] 2 triangle2[Role,CourseSession,Instructor,r] 2
 RHS1 triangle2[Role,CourseSession,Instructor,c] 2 triangle2[Role,Clinic,Location,r] 2
 RHS1 triangle2[Role,Clinic,Location,c] 2 triangle2[Role,Clinic,Therapist,r] 2
 RHS1 triangle2[Role,Clinic,Therapist,c] 2 triangle2[Role,Clinic,Instructor,r] 2
 RHS1 triangle2[Role,Clinic,Instructor,c] 2 triangle2[Role,Location,Therapist,r] 2
 RHS1 triangle2[Role,Location,Therapist,c] 2 triangle2[Role,Location,Instructor,r] 2
 RHS1 triangle2[Role,Location,Instructor,c] 2 triangle2[Role,Therapist,Instructor,r] 2
 RHS1 triangle2[Role,Therapist,Instructor,c] 2 triangle2[User,CourseSession,Clinic,r] 2
 RHS1 triangle2[User,CourseSession,Clinic,c] 2 triangle2[User,CourseSession,Location,r] 2
 RHS1 triangle2[User,CourseSession,Location,c] 2 triangle2[User,CourseSession,Therapist,r] 2
 RHS1 triangle2[User,CourseSession,Therapist,c] 2 triangle2[User,CourseSession,Instructor,r] 2
 RHS1 triangle2[User,CourseSession,Instructor,c] 2 triangle2[User,Clinic,Location,r] 2
 RHS1 triangle2[User,Clinic,Location,c] 2 triangle2[User,Clinic,Therapist,r] 2
 RHS1 triangle2[User,Clinic,Therapist,c] 2 triangle2[User,Clinic,Instructor,r] 2
 RHS1 triangle2[User,Clinic,Instructor,c] 2 triangle2[User,Location,Therapist,r] 2
 RHS1 triangle2[User,Location,Therapist,c] 2 triangle2[User,Location,Instructor,r] 2
 RHS1 triangle2[User,Location,Instructor,c] 2 triangle2[User,Therapist,Instructor,r] 2
 RHS1 triangle2[User,Therapist,Instructor,c] 2 triangle2[CourseSession,Clinic,Location,r] 2
 RHS1 triangle2[CourseSession,Clinic,Location,c] 2 triangle2[CourseSession,Clinic,Therapist,r] 2
 RHS1 triangle2[CourseSession,Clinic,Therapist,c] 2 triangle2[CourseSession,Clinic,Instructor,r] 2
 RHS1 triangle2[CourseSession,Clinic,Instructor,c] 2 triangle2[CourseSession,Location,Therapist,r] 2
 RHS1 triangle2[CourseSession,Location,Therapist,c] 2 triangle2[CourseSession,Location,Instructor,r] 2
 RHS1 triangle2[CourseSession,Location,Instructor,c] 2 triangle2[CourseSession,Therapist,Instructor,r] 2
 RHS1 triangle2[CourseSession,Therapist,Instructor,c] 2 triangle2[Clinic,Location,Therapist,r] 2
 RHS1 triangle2[Clinic,Location,Therapist,c] 2 triangle2[Clinic,Location,Instructor,r] 2
 RHS1 triangle2[Clinic,Location,Instructor,c] 2 triangle2[Clinic,Therapist,Instructor,r] 2
 RHS1 triangle2[Clinic,Therapist,Instructor,c] 2 triangle2[Location,Therapist,Instructor,r] 2
 RHS1 triangle2[Location,Therapist,Instructor,c] 2
BOUNDS
 LO BND1 x[r,Client] 1
 UP BND1 x[r,Client] 34
 LO BND1 x[r,CertificateTemplate] 1
 UP BND1 x[r,CertificateTemplate] 36
 LO BND1 x[r,Course] 1
 UP BND1 x[r,Course] 37
 LO BND1 x[r,Role] 1
 UP BND1 x[r,Role] 37
 LO BND1 x[r,User] 1
 UP BND1 x[r,User] 36
 LO BND1 x[r,CourseSession] 1
 UP BND1 x[r,CourseSession] 34
 LO BND1 x[r,Clinic] 1
 UP BND1 x[r,Clinic] 35
 LO BND1 x[r,Location] 1
 UP BND1 x[r,Location] 34
 LO BND1 x[r,Therapist] 1
 UP BND1 x[r,Therapist] 33
 LO BND1 x[r,Instructor] 1
 UP BND1 x[r,Instructor] 33
 LO BND1 x[c,Client] 1
 UP BND1 x[c,Client] 32
 LO BND1 x[c,CertificateTemplate] 1
 UP BND1 x[c,CertificateTemplate] 31
 LO BND1 x[c,Course] 1
 UP BND1 x[c,Course] 33
 LO BND1 x[c,Role] 1
 UP BND1 x[c,Role] 33
 LO BND1 x[c,User] 1
 UP BND1 x[c,User] 32
 LO BND1 x[c,CourseSession] 1
 UP BND1 x[c,CourseSession] 29
 LO BND1 x[c,Clinic] 1
 UP BND1 x[c,Clinic] 32
 LO BND1 x[c,Location] 1
 UP BND1 x[c,Location] 32
 LO BND1 x[c,Therapist] 1
 UP BND1 x[c,Therapist] 32
 LO BND1 x[c,Instructor] 1
 UP BND1 x[c,Instructor] 31
 FR BND1 center_dist[r,Client]
 FR BND1 center_dist[r,CertificateTemplate]
 FR BND1 center_dist[r,Course]
 FR BND1 center_dist[r,Role]
 FR BND1 center_dist[r,User]
 FR BND1 center_dist[r,CourseSession]
 FR BND1 center_dist[r,Clinic]
 FR BND1 center_dist[r,Location]
 FR BND1 center_dist[r,Therapist]
 FR BND1 center_dist[r,Instructor]
 FR BND1 center_dist[c,Client]
 FR BND1 center_dist[c,CertificateTemplate]
 FR BND1 center_dist[c,Course]
 FR BND1 center_dist[c,Role]
 FR BND1 center_dist[c,User]
 FR BND1 center_dist[c,CourseSession]
 FR BND1 center_dist[c,Clinic]
 FR BND1 center_dist[c,Location]
 FR BND1 center_dist[c,Therapist]
 FR BND1 center_dist[c,Instructor]
 UP BND1 dist[r,Client,CertificateTemplate] 43
 UP BND1 dist[r,Client,CourseSession] 42
 UP BND1 dist[r,Client,Instructor] 42.5
 UP BND1 dist[r,Client,Location] 42
 UP BND1 dist[r,Client,User] 43
 UP BND1 dist[r,CertificateTemplate,CourseSession] 43
 UP BND1 dist[r,Course,CourseSession] 43.5
 UP BND1 dist[r,CourseSession,Instructor] 42.5
 UP BND1 dist[r,Instructor,Location] 42.5
 UP BND1 dist[r,Clinic,Location] 42.5
 UP BND1 dist[r,Clinic,Therapist] 43
 UP BND1 dist[r,Location,Therapist] 42.5
 UP BND1 dist[r,Location,CourseSession] 42
 UP BND1 dist[r,Therapist,CourseSession] 42.5
 UP BND1 dist[r,Role,User] 42.5
 UP BND1 dist[r,User,Instructor] 43.5
 UP BND1 dist[r,User,Therapist] 43.5
 UP BND1 dist[c,Client,CertificateTemplate] 42.5
 UP BND1 dist[c,Client,CourseSession] 43.5
 UP BND1 dist[c,Client,Instructor] 42.5
 UP BND1 dist[c,Client,Location] 42
 UP BND1 dist[c,Client,User] 42
 UP BND1 dist[c,CertificateTemplate,CourseSession] 43
 UP BND1 dist[c,Course,CourseSession] 44
 UP BND1 dist[c,CourseSession,Instructor] 43
 UP BND1 dist[c,Instructor,Location] 42.5
 UP BND1 dist[c,Clinic,Location] 42
 UP BND1 dist[c,Clinic,Therapist] 42
 UP BND1 dist[c,Location,Therapist] 42
 UP BND1 dist[c,Location,CourseSession] 43.5
 UP BND1 dist[c,Therapist,CourseSession] 43.5
 UP BND1 dist[c,Role,User] 42.5
 UP BND1 dist[c,User,Instructor] 42.5
 UP BND1 dist[c,User,Therapist] 42
 UP BND1 zV[r,rt,Client,CertificateTemplate] 1
 UP BND1 zV[c,rt,Client,CertificateTemplate] 1
 UP BND1 zV[r,lb,Client,CertificateTemplate] 1
 UP BND1 zV[c,lb,Client,CertificateTemplate] 1
 UP BND1 zV[r,rt,Client,Course] 1
 UP BND1 zV[c,rt,Client,Course] 1
 UP BND1 zV[r,lb,Client,Course] 1
 UP BND1 zV[c,lb,Client,Course] 1
 UP BND1 zV[r,rt,Client,Role] 1
 UP BND1 zV[c,rt,Client,Role] 1
 UP BND1 zV[r,lb,Client,Role] 1
 UP BND1 zV[c,lb,Client,Role] 1
 UP BND1 zV[r,rt,Client,User] 1
 UP BND1 zV[c,rt,Client,User] 1
 UP BND1 zV[r,lb,Client,User] 1
 UP BND1 zV[c,lb,Client,User] 1
 UP BND1 zV[r,rt,Client,CourseSession] 1
 UP BND1 zV[c,rt,Client,CourseSession] 1
 UP BND1 zV[r,lb,Client,CourseSession] 1
 UP BND1 zV[c,lb,Client,CourseSession] 1
 UP BND1 zV[r,rt,Client,Clinic] 1
 UP BND1 zV[c,rt,Client,Clinic] 1
 UP BND1 zV[r,lb,Client,Clinic] 1
 UP BND1 zV[c,lb,Client,Clinic] 1
 UP BND1 zV[r,rt,Client,Location] 1
 UP BND1 zV[c,rt,Client,Location] 1
 UP BND1 zV[r,lb,Client,Location] 1
 UP BND1 zV[c,lb,Client,Location] 1
 UP BND1 zV[r,rt,Client,Therapist] 1
 UP BND1 zV[c,rt,Client,Therapist] 1
 UP BND1 zV[r,lb,Client,Therapist] 1
 UP BND1 zV[c,lb,Client,Therapist] 1
 UP BND1 zV[r,rt,Client,Instructor] 1
 UP BND1 zV[c,rt,Client,Instructor] 1
 UP BND1 zV[r,lb,Client,Instructor] 1
 UP BND1 zV[c,lb,Client,Instructor] 1
 UP BND1 zV[r,rt,CertificateTemplate,Course] 1
 UP BND1 zV[c,rt,CertificateTemplate,Course] 1
 UP BND1 zV[r,lb,CertificateTemplate,Course] 1
 UP BND1 zV[c,lb,CertificateTemplate,Course] 1
 UP BND1 zV[r,rt,CertificateTemplate,Role] 1
 UP BND1 zV[c,rt,CertificateTemplate,Role] 1
 UP BND1 zV[r,lb,CertificateTemplate,Role] 1
 UP BND1 zV[c,lb,CertificateTemplate,Role] 1
 UP BND1 zV[r,rt,CertificateTemplate,User] 1
 UP BND1 zV[c,rt,CertificateTemplate,User] 1
 UP BND1 zV[r,lb,CertificateTemplate,User] 1
 UP BND1 zV[c,lb,CertificateTemplate,User] 1
 UP BND1 zV[r,rt,CertificateTemplate,CourseSession] 1
 UP BND1 zV[c,rt,CertificateTemplate,CourseSession] 1
 UP BND1 zV[r,lb,CertificateTemplate,CourseSession] 1
 UP BND1 zV[c,lb,CertificateTemplate,CourseSession] 1
 UP BND1 zV[r,rt,CertificateTemplate,Clinic] 1
 UP BND1 zV[c,rt,CertificateTemplate,Clinic] 1
 UP BND1 zV[r,lb,CertificateTemplate,Clinic] 1
 UP BND1 zV[c,lb,CertificateTemplate,Clinic] 1
 UP BND1 zV[r,rt,CertificateTemplate,Location] 1
 UP BND1 zV[c,rt,CertificateTemplate,Location] 1
 UP BND1 zV[r,lb,CertificateTemplate,Location] 1
 UP BND1 zV[c,lb,CertificateTemplate,Location] 1
 UP BND1 zV[r,rt,CertificateTemplate,Therapist] 1
 UP BND1 zV[c,rt,CertificateTemplate,Therapist] 1
 UP BND1 zV[r,lb,CertificateTemplate,Therapist] 1
 UP BND1 zV[c,lb,CertificateTemplate,Therapist] 1
 UP BND1 zV[r,rt,CertificateTemplate,Instructor] 1
 UP BND1 zV[c,rt,CertificateTemplate,Instructor] 1
 UP BND1 zV[r,lb,CertificateTemplate,Instructor] 1
 UP BND1 zV[c,lb,CertificateTemplate,Instructor] 1
 UP BND1 zV[r,rt,Course,Role] 1
 UP BND1 zV[c,rt,Course,Role] 1
 UP BND1 zV[r,lb,Course,Role] 1
 UP BND1 zV[c,lb,Course,Role] 1
 UP BND1 zV[r,rt,Course,User] 1
 UP BND1 zV[c,rt,Course,User] 1
 UP BND1 zV[r,lb,Course,User] 1
 UP BND1 zV[c,lb,Course,User] 1
 UP BND1 zV[r,rt,Course,CourseSession] 1
 UP BND1 zV[c,rt,Course,CourseSession] 1
 UP BND1 zV[r,lb,Course,CourseSession] 1
 UP BND1 zV[c,lb,Course,CourseSession] 1
 UP BND1 zV[r,rt,Course,Clinic] 1
 UP BND1 zV[c,rt,Course,Clinic] 1
 UP BND1 zV[r,lb,Course,Clinic] 1
 UP BND1 zV[c,lb,Course,Clinic] 1
 UP BND1 zV[r,rt,Course,Location] 1
 UP BND1 zV[c,rt,Course,Location] 1
 UP BND1 zV[r,lb,Course,Location] 1
 UP BND1 zV[c,lb,Course,Location] 1
 UP BND1 zV[r,rt,Course,Therapist] 1
 UP BND1 zV[c,rt,Course,Therapist] 1
 UP BND1 zV[r,lb,Course,Therapist] 1
 UP BND1 zV[c,lb,Course,Therapist] 1
 UP BND1 zV[r,rt,Course,Instructor] 1
 UP BND1 zV[c,rt,Course,Instructor] 1
 UP BND1 zV[r,lb,Course,Instructor] 1
 UP BND1 zV[c,lb,Course,Instructor] 1
 UP BND1 zV[r,rt,Role,User] 1
 UP BND1 zV[c,rt,Role,User] 1
 UP BND1 zV[r,lb,Role,User] 1
 UP BND1 zV[c,lb,Role,User] 1
 UP BND1 zV[r,rt,Role,CourseSession] 1
 UP BND1 zV[c,rt,Role,CourseSession] 1
 UP BND1 zV[r,lb,Role,CourseSession] 1
 UP BND1 zV[c,lb,Role,CourseSession] 1
 UP BND1 zV[r,rt,Role,Clinic] 1
 UP BND1 zV[c,rt,Role,Clinic] 1
 UP BND1 zV[r,lb,Role,Clinic] 1
 UP BND1 zV[c,lb,Role,Clinic] 1
 UP BND1 zV[r,rt,Role,Location] 1
 UP BND1 zV[c,rt,Role,Location] 1
 UP BND1 zV[r,lb,Role,Location] 1
 UP BND1 zV[c,lb,Role,Location] 1
 UP BND1 zV[r,rt,Role,Therapist] 1
 UP BND1 zV[c,rt,Role,Therapist] 1
 UP BND1 zV[r,lb,Role,Therapist] 1
 UP BND1 zV[c,lb,Role,Therapist] 1
 UP BND1 zV[r,rt,Role,Instructor] 1
 UP BND1 zV[c,rt,Role,Instructor] 1
 UP BND1 zV[r,lb,Role,Instructor] 1
 UP BND1 zV[c,lb,Role,Instructor] 1
 UP BND1 zV[r,rt,User,CourseSession] 1
 UP BND1 zV[c,rt,User,CourseSession] 1
 UP BND1 zV[r,lb,User,CourseSession] 1
 UP BND1 zV[c,lb,User,CourseSession] 1
 UP BND1 zV[r,rt,User,Clinic] 1
 UP BND1 zV[c,rt,User,Clinic] 1
 UP BND1 zV[r,lb,User,Clinic] 1
 UP BND1 zV[c,lb,User,Clinic] 1
 UP BND1 zV[r,rt,User,Location] 1
 UP BND1 zV[c,rt,User,Location] 1
 UP BND1 zV[r,lb,User,Location] 1
 UP BND1 zV[c,lb,User,Location] 1
 UP BND1 zV[r,rt,User,Therapist] 1
 UP BND1 zV[c,rt,User,Therapist] 1
 UP BND1 zV[r,lb,User,Therapist] 1
 UP BND1 zV[c,lb,User,Therapist] 1
 UP BND1 zV[r,rt,User,Instructor] 1
 UP BND1 zV[c,rt,User,Instructor] 1
 UP BND1 zV[r,lb,User,Instructor] 1
 UP BND1 zV[c,lb,User,Instructor] 1
 UP BND1 zV[r,rt,CourseSession,Clinic] 1
 UP BND1 zV[c,rt,CourseSession,Clinic] 1
 UP BND1 zV[r,lb,CourseSession,Clinic] 1
 UP BND1 zV[c,lb,CourseSession,Clinic] 1
 UP BND1 zV[r,rt,CourseSession,Location] 1
 UP BND1 zV[c,rt,CourseSession,Location] 1
 UP BND1 zV[r,lb,CourseSession,Location] 1
 UP BND1 zV[c,lb,CourseSession,Location] 1
 UP BND1 zV[r,rt,CourseSession,Therapist] 1
 UP BND1 zV[c,rt,CourseSession,Therapist] 1
 UP BND1 zV[r,lb,CourseSession,Therapist] 1
 UP BND1 zV[c,lb,CourseSession,Therapist] 1
 UP BND1 zV[r,rt,CourseSession,Instructor] 1
 UP BND1 zV[c,rt,CourseSession,Instructor] 1
 UP BND1 zV[r,lb,CourseSession,Instructor] 1
 UP BND1 zV[c,lb,CourseSession,Instructor] 1
 UP BND1 zV[r,rt,Clinic,Location] 1
 UP BND1 zV[c,rt,Clinic,Location] 1
 UP BND1 zV[r,lb,Clinic,Location] 1
 UP BND1 zV[c,lb,Clinic,Location] 1
 UP BND1 zV[r,rt,Clinic,Therapist] 1
 UP BND1 zV[c,rt,Clinic,Therapist] 1
 UP BND1 zV[r,lb,Clinic,Therapist] 1
 UP BND1 zV[c,lb,Clinic,Therapist] 1
 UP BND1 zV[r,rt,Clinic,Instructor] 1
 UP BND1 zV[c,rt,Clinic,Instructor] 1
 UP BND1 zV[r,lb,Clinic,Instructor] 1
 UP BND1 zV[c,lb,Clinic,Instructor] 1
 UP BND1 zV[r,rt,Location,Therapist] 1
 UP BND1 zV[c,rt,Location,Therapist] 1
 UP BND1 zV[r,lb,Location,Therapist] 1
 UP BND1 zV[c,lb,Location,Therapist] 1
 UP BND1 zV[r,rt,Location,Instructor] 1
 UP BND1 zV[c,rt,Location,Instructor] 1
 UP BND1 zV[r,lb,Location,Instructor] 1
 UP BND1 zV[c,lb,Location,Instructor] 1
 UP BND1 zV[r,rt,Therapist,Instructor] 1
 UP BND1 zV[c,rt,Therapist,Instructor] 1
 UP BND1 zV[r,lb,Therapist,Instructor] 1
 UP BND1 zV[c,lb,Therapist,Instructor] 1
ENDATA
