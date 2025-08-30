// Role configuration and dynamic UI logic extracted from inline script
const roles = {
  super_admin: { id:'super_admin', name:'Super Administrator', description:'You have full access to all facilities and patients.', badgeClass:'super-admin-badge', badgeText:'Super Admin', userName:'Alex Johnson (Super Admin)', facilityAccess:'all', patientAccess:'all' },
  facility_admin: { id:'facility_admin', name:'Facility Administrator', description:'You have access to all patients in Facility 1.', badgeClass:'facility-admin-badge', badgeText:'Facility Admin', userName:'Sarah Miller (Facility 1 Admin)', facilityId:1, patientAccess:'facility' },
  staff: { id:'staff', name:'Staff Provider', description:'You have access only to your assigned patients in Facility 1.', badgeClass:'staff-badge', badgeText:'Staff Provider', userName:'James Wilson (Staff Provider)', facilityId:1, assignedPatientIds:[1,3,5] }
};
let currentUser = roles.super_admin;
function switchRole(roleId){ currentUser = roles[roleId]; updateUIForCurrentRole(); }
function updateUIForCurrentRole(){
  const roleBadge = document.getElementById('roleBadge'); if(roleBadge){ roleBadge.className = `role-badge ${currentUser.badgeClass}`; roleBadge.textContent = currentUser.badgeText; }
  const roleDesc = document.getElementById('roleDescription'); if(roleDesc) roleDesc.textContent = currentUser.description;
  const userNameEl = document.getElementById('currentUserName'); if(userNameEl) userNameEl.textContent = currentUser.userName;
  const staffWarning = document.getElementById('staffWarning'); if(staffWarning){ currentUser.id==='staff'?staffWarning.classList.remove('d-none'):staffWarning.classList.add('d-none'); }
  filterDataBasedOnRole();
}
function filterDataBasedOnRole(){ let dataStatus=''; switch(currentUser.id){ case 'super_admin': dataStatus='Showing data for ALL facilities and patients'; break; case 'facility_admin': dataStatus=`Showing data for Facility ${currentUser.facilityId} (all patients)`; break; case 'staff': dataStatus=`Showing data for your assigned patients: ${currentUser.assignedPatientIds.join(', ')}`; break;} console.log('Data filtering applied:', dataStatus); }
// Theme handling
function initTheme(){ const themeSwitch=document.getElementById('themeSwitch'); const themeSelect=document.getElementById('themeSelect'); if(localStorage.theme==='dark'){ document.body.classList.add('dark-theme'); if(themeSwitch) themeSwitch.checked=true; if(themeSelect) themeSelect.value='dark'; }
  if(themeSwitch){ themeSwitch.onchange=e=>{ const dark=e.target.checked; document.body.classList.toggle('dark-theme', dark); if(themeSelect) themeSelect.value=dark?'dark':'light'; localStorage.theme=dark?'dark':'light'; }; }
  if(themeSelect){ themeSelect.onchange=e=>{ const dark=e.target.value==='dark'; document.body.classList.toggle('dark-theme', dark); if(themeSwitch) themeSwitch.checked=dark; localStorage.theme=e.target.value; }; }
}
// Avatar upload feedback
function initAvatarUpload(){ const avatarUpload=document.getElementById('avatarUpload'); if(!avatarUpload) return; avatarUpload.onchange=()=>{ const file=avatarUpload.files[0]; if(!file) return; alert('Avatar updated successfully!'); }; }
// Charts
function initCharts(){ if(typeof Chart==='undefined') return; new Chart(document.getElementById('healthDonut'), { type:'doughnut', data:{ labels:['Mood','Activity','Sleep','Environment'], datasets:[{ data:[35,25,25,15], backgroundColor:['#4e73df','#1cc88a','#36b9cc','#f6c23e'], borderWidth:0 }]}, options:{ cutout:'75%', plugins:{ legend:{display:false}, tooltip:{ callbacks:{ label: ctx=>`${ctx.label}: ${ctx.parsed}%` } } }, responsive:true, maintainAspectRatio:false } });
  const moodCtx=document.getElementById('moodChart'); if(moodCtx){ new Chart(moodCtx.getContext('2d'), { type:'line', data:{ labels:['Mon','Tue','Wed','Thu','Fri','Sat','Sun'], datasets:[{ label:'Mood Level', data:[3.2,4.1,3.8,4.5,4.8,5.2,4.9], borderColor:'#4e73df', backgroundColor:'rgba(78,115,223,0.05)', tension:0.4, fill:true, pointBackgroundColor:'#fff', pointBorderColor:'#4e73df', pointBorderWidth:2, pointRadius:4 }]}, options:{ responsive:true, maintainAspectRatio:false, scales:{ y:{ min:1, max:6, ticks:{ callback:value=>['','Sad','Low','Neutral','Fair','Good','Happy'][value] } } }, plugins:{ legend:{display:false}, tooltip:{ callbacks:{ label:ctx=>`Mood: ${['','Sad','Low','Neutral','Fair','Good','Happy'][ctx.parsed.y]}` } } } } }); }
  const moveCtx=document.getElementById('moveChart'); if(moveCtx){ new Chart(moveCtx.getContext('2d'), { type:'bar', data:{ labels:['Mon','Tue','Wed','Thu','Fri','Sat','Sun'], datasets:[{ label:'Steps (thousands)', data:[7.2,8.5,6.8,9.2,8.4,10.1,7.9], backgroundColor:'#1cc88a', borderRadius:5 }]}, options:{ responsive:true, maintainAspectRatio:false, scales:{ y:{ beginAtZero:true, title:{ display:true, text:'Steps (thousands)'} } }, plugins:{ legend:{display:false} } } }); }
}
// Buttons & interactions
function initButtons(){ const syncBtn=document.getElementById('syncWearable'); if(syncBtn){ syncBtn.addEventListener('click', function(){ const btn=this; btn.innerHTML='<i class="fas fa-spinner fa-spin me-2"></i> Syncing...'; btn.disabled=true; setTimeout(()=>{ btn.innerHTML='<i class="fas fa-check-circle me-2"></i> Synced!'; setTimeout(()=>{ btn.innerHTML='<i class="fas fa-sync-alt me-2"></i> Sync Devices Now'; btn.disabled=false; },2000); },1500); }); }
  const addGoal=document.getElementById('addGoalBtn'); if(addGoal){ addGoal.addEventListener('click', ()=>{ const goalName=prompt('Enter your new goal:'); if(goalName) alert(`Goal "${goalName}" added successfully!`); }); }
}
// Init
document.addEventListener('DOMContentLoaded', () => { initTheme(); initAvatarUpload(); initCharts(); updateUIForCurrentRole(); initButtons(); });
